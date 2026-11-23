"""
The visual guide to the project: docs/visual_guide.html, one self-contained page of illustrated scenes
for people new to the project. Every number and every trace in it is read from the project's own
results when this script runs, so the page is regenerated, never edited by hand.

    uv run python visual_guide.py          (from experiments/student)

Reads: cache (one HMC night, one CinC night), results/detections(_clean), results/benchmark,
results/student/{teachers,consensus,injection,work/runs}.
"""
import html
import json

import numpy as np
import pandas as pd

import data
import label_models
from spindlebench import paths, provenance

OUT = paths.ROOT / "docs" / "visual_guide.html"
T = list(data.TEACHERS)
NAME = {"a7": "A7", "luna_spindles": "Luna", "wonambi_ucsd": "UCSD", "yasa_default": "YASA"}
CSS_ID = {"a7": "a7", "luna_spindles": "luna", "wonambi_ucsd": "ucsd", "yasa_default": "yasa"}
PRETTY = {"sumov2": "SUMOv2", "s5": "S5", "a7": "A7", "wonambi_moelle2011": "Mölle (Wonambi)",
          "yasa_dreams_tuned": "YASA, DREAMS-tuned", "openspindlenet": "OpenSpindleNet", "wonambi_ucsd": "UCSD",
          "luna_spindles": "Luna", "sumo": "SUMO"}
MARK = {"a7": "circle", "luna_spindles": "triangle", "wonambi_ucsd": "square", "yasa_default": "diamond"}


def esc(s):
    return html.escape(str(s))


def marker(kind, x, y, r=5, cls=""):
    if kind == "circle":
        return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if kind == "square":
        return f'<rect class="{cls}" x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1"/>'
    if kind == "triangle":
        return f'<path class="{cls}" d="M{x:.1f},{y - r * 1.15:.1f} L{x + r:.1f},{y + r * 0.85:.1f} L{x - r:.1f},{y + r * 0.85:.1f} Z"/>'
    return f'<path class="{cls}" d="M{x:.1f},{y - r * 1.2:.1f} L{x + r * 1.05:.1f},{y:.1f} L{x:.1f},{y + r * 1.2:.1f} L{x - r * 1.05:.1f},{y:.1f} Z"/>'


# ------------------------------------------------------------------ data for the scenes

def crop_scene():
    """the 40-s HMC SN001 crop of docs/figures (valid N2, 3+ consensus events in the central 30 s)"""
    x, code = data.open_night("hmc", "SN001")
    code = np.asarray(code)
    valid = (code & data.VALID_BIT) > 0
    mv = label_models.load_targets()["mv"][code & (data.VALID_BIT - 1)] > 0.5
    for a in range(0, len(code) - data.CROP, 500):
        if valid[a:a + data.CROP].all():
            r = data.mask_to_events(mv[a:a + data.CROP])
            if len(r) >= 3 and r[:, 0].min() >= data.CORE_LO and r[:, 1].max() <= data.CORE_LO + data.CORE:
                break
    t0 = a / data.FS
    sig = np.asarray(x[a:a + data.CROP], float)
    ev = {}
    for t in T:
        on, du = data.load_teacher_events("hmc", "SN001", t, "C3-M2")
        k = (on + du > t0) & (on < t0 + 40)
        ev[t] = [(max(o - t0, 0), min(o + d - t0, 40)) for o, d in zip(on[k], du[k])]
    return dict(t0=t0, sig=sig, ev=ev, mv=[(s / data.FS, e / data.FS) for s, e in r])


def hours():
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    h = m.groupby("dataset").duration.sum() / 3600
    return {k: float(h.get(k, 0)) for k in ("dreams_spindles", "nap", "hmc", "cinc2018")}


def leaderboard():
    s = pd.read_csv(paths.RESULTS / "benchmark" / "summary.csv")
    s = s[(s.event == "spindle") & (s.scope == "N2") & (s.tol == 0.5)]
    d = s[(s.dataset == "dreams_spindles") & (s.reference == "expert1") & (s.rule == "onset")]
    n = s[(s.dataset == "nap") & (s.reference == "expert") & (s.rule == "center")]
    d = d[~d.detector.isin(["sumov2_n2", "a7_n2base", "a7_incontext_n2base", "dreams_auto", "yasa_dreams_tuned"])]
    n = n[~n.detector.isin(["sumov2_n2", "a7_n2base", "a7_incontext_n2base"])]
    top = lambda q: [(r.detector, float(r.F1_pooled)) for r in q.sort_values("F1_pooled", ascending=False).head(3).itertuples()]
    return dict(dreams=top(d), nap=top(n), n_detectors=int(s.detector.nunique()))


def consensus():
    s = pd.read_csv(data.STUDY / "consensus" / "summary_partial.csv")
    t = pd.read_csv(data.STUDY / "consensus_teachers" / "summary_partial.csv")
    out = {}
    for ds, ref, rule in (("dreams_spindles", "expert1", "onset"), ("nap", "expert", "center")):
        def q(f, ds=ds, ref=ref, rule=rule):
            k = (f.dataset == ds) & (f.reference == ref) & (f.rule == rule) & (f.tol == 0.5) & \
                (f.scope == "N2") & (f.event == "spindle")
            return f[k].set_index("detector").F1_pooled
        out[ds] = dict(mv=float(q(s)["consensus_mv"]), teachers={k: float(v) for k, v in q(t).items()})
    return out


def artefact_case(rec="tr12-0209"):
    h = pd.read_csv(paths.CACHE / "labels" / "cinc2018" / f"{rec}__hypno.csv")
    a = pd.read_csv(paths.CACHE / "labels" / "cinc2018" / f"{rec}__artefact.csv")
    counts = {}
    for t in ("luna_spindles", "wonambi_ucsd"):
        b = len(pd.read_csv(paths.RESULTS / "detections" / "cinc2018" / t / f"{rec}.csv"))
        c = len(pd.read_csv(paths.RESULTS / "detections_clean" / "cinc2018" / t / f"{rec}.csv"))
        counts[t] = (b, c)
    return dict(rec=rec, stages=h.stage.tolist(), epoch=float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0]),
                flagged=a.loc[a.artefact == 1, ["start", "end"]].to_numpy().tolist(),
                dur=float(a.end.iloc[-1]), counts=counts, share=float(a.artefact.mean()))


def teacher_cards():
    d = pd.read_csv(data.STUDY / "teachers" / "density.csv").groupby("teacher").N2_per_min.mean()
    r = pd.read_csv(data.STUDY / "injection" / "teacher_recall.csv")
    fa = pd.read_csv(data.STUDY / "injection" / "teacher_false_alarms.csv").set_index("teacher").per_quiet_min
    lm = json.loads((data.WORK / "targets" / "label_models.json").read_text())["info"]["ds"]["teachers"]
    rec9 = r[r.snr_db == 9.0].set_index("teacher").recall
    return {t: dict(n2=float(d[t]), rec9=float(rec9[t]), fa=float(fa[t]), ds_recall=float(lm[t]["recall"]),
                    ds_spec=float(lm[t]["specificity"])) for t in T}


def patterns():
    t = pd.read_csv(data.STUDY / "teachers" / "label_models.csv")
    return t


def recall_curves():
    r = pd.read_csv(data.STUDY / "injection" / "teacher_recall.csv")
    return r


def agreement():
    a = pd.read_csv(data.STUDY / "teachers" / "agreement.csv")
    return float(a.f1.min()), float(a.f1.max())


def pilot():
    f = data.RUNS / "student_soft_s0" / "metrics.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    return dict(passes=int(m["pass"].max()), f1_mv=float(m.f1_mv.iloc[-1]), best=float(m.f1_mv.max()),
                done=(f.parent / "done.json").exists())


# ------------------------------------------------------------------ SVG scenes

W = 960


def svg(h, body, label):
    return (f'<figure class="scene-fig"><svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(label)}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg></figure>')


def scene_spindle(c):
    """the real crop: EEG, four teacher rows, the vote"""
    x0, x1 = 120, 930
    sx = lambda t: x0 + (x1 - x0) * t / 40
    sig = c["sig"][::2]
    y = 105 - 26 * np.clip(sig, -3.5, 3.5)
    pts = " ".join(f"{sx(i * 0.02):.1f},{v:.1f}" for i, v in enumerate(y))
    b = []
    for s, e in c["mv"]:
        b.append(f'<rect class="hl" x="{sx(s):.1f}" y="18" width="{sx(e) - sx(s):.1f}" height="370" rx="3"/>')
    b.append('<text class="lbl" x="20" y="100">EEG</text><text class="lbl-s" x="20" y="116">C3-M2, 40 s</text>')
    b.append(f'<polyline class="eeg" points="{pts}"/>')
    for k, t in enumerate(T):
        yy = 205 + 38 * k
        b.append(f'<line class="row" x1="{x0}" x2="{x1}" y1="{yy}" y2="{yy}"/>')
        b.append(marker(MARK[t], 30, yy, 6, f"f-{CSS_ID[t]}"))
        b.append(f'<text class="lbl" x="44" y="{yy + 5}">{NAME[t]}</text>')
        for s, e in c["ev"][t]:
            b.append(f'<rect class="f-{CSS_ID[t]}" x="{sx(s):.1f}" y="{yy - 9}" width="{max(sx(e) - sx(s), 3):.1f}" height="18" rx="9"/>')
    yv = 205 + 38 * 4 + 12
    b.append(f'<text class="lbl strong" x="20" y="{yv + 5}">vote ≥ 2</text>')
    for s, e in c["mv"]:
        b.append(f'<rect class="f-ink" x="{sx(s):.1f}" y="{yv - 10}" width="{sx(e) - sx(s):.1f}" height="20" rx="10"/>')
    for t in range(0, 41, 5):
        b.append(f'<text class="tick" x="{sx(t):.1f}" y="408" text-anchor="middle">{t} s</text>')
    s0 = c["mv"][0][0]
    b.append(f'<text class="note" x="{sx(s0) - 8:.1f}" y="34" text-anchor="end">a spindle the teachers agree on →</text>')
    return svg(420, "".join(b), "A real 40-second EEG crop with the four teachers' detections and their majority vote")


def scene_hours(h):
    tot = sum(h.values())
    x0, x1 = 40, 920
    b = []
    y = 70
    cx = x0
    parts = [("dreams_spindles", "DREAMS", "gold"), ("nap", "NAP", "gold"), ("hmc", "HMC", "free"),
             ("cinc2018", "CinC 2018", "free")]
    for k, name, kind in parts:
        w = (x1 - x0) * h[k] / tot
        cls = "gold-fill" if kind == "gold" else "free-fill"
        b.append(f'<rect class="{cls}" x="{cx:.1f}" y="{y}" width="{max(w - 2, 1.5):.1f}" height="64" rx="4"/>')
        if kind == "free":
            b.append(f'<text class="lbl-in" x="{cx + 14:.1f}" y="{y + 30}">{name}</text>'
                     f'<text class="lbl-in-s" x="{cx + 14:.1f}" y="{y + 50}">{h[k]:,.0f} h · sleep stages only</text>')
        cx += w
    g = h["dreams_spindles"] + h["nap"]
    b.append(f'<path class="leader" d="M{x0 + 3},{y + 66} L{x0 + 3},{y + 104} L{x0 + 28},{y + 104}"/>')
    b.append(f'<text class="lbl strong" x="{x0 + 36}" y="{y + 109}">DREAMS {h["dreams_spindles"]:.0f} h + NAP {h["nap"]:.0f} h '
             f'= {g:.0f} h with expert spindle marks</text>')
    b.append(f'<text class="lbl-s" x="{x0 + 36}" y="{y + 129}">{100 * g / tot:.1f}% of the EEG in this project. '
             f'These two are kept for scoring only.</text>')
    b.append(f'<text class="lbl" x="{x0}" y="40">All recordings used, drawn to scale: {tot:,.0f} hours</text>')
    return svg(220, "".join(b), "Hours of expert-labelled versus unlabelled EEG, drawn to scale")


def scene_benchmark(lb):
    b = []
    # stations
    st = [(30, "datasets", "DREAMS · NAP · MASS<br/>BOAS · HMC · CinC"), (230, "one cache", "same format<br/>for every detector"),
          (430, f"{lb['n_detectors']} detector set-ups", "rule-based and<br/>learned, each wrapped"),
          (630, "one referee", "same matching rule,<br/>same expert, same stage")]
    for x, title, sub in st:
        b.append(f'<rect class="card" x="{x}" y="30" width="170" height="96" rx="14"/>')
        b.append(f'<text class="lbl strong" x="{x + 85}" y="64" text-anchor="middle">{title}</text>')
        for i, line in enumerate(sub.split("<br/>")):
            b.append(f'<text class="lbl-s" x="{x + 85}" y="{88 + 17 * i}" text-anchor="middle">{line}</text>')
    for x in (200, 400, 600):
        b.append(f'<path class="arrow" d="M{x + 4},78 L{x + 26},78" marker-end="url(#ah)"/>')
    b.append('<path class="arrow" d="M804,78 L830,78" marker-end="url(#ah)"/>')
    # leaderboard
    b.append('<rect class="card strong-card" x="836" y="14" width="114" height="128" rx="14"/>')
    b.append('<text class="lbl strong" x="893" y="50" text-anchor="middle">leader-</text>'
             '<text class="lbl strong" x="893" y="70" text-anchor="middle">board</text>'
             '<text class="lbl-s" x="893" y="96" text-anchor="middle">RQ1–RQ4</text>')
    # podiums
    for j, (key, title) in enumerate((("dreams", "DREAMS (vs expert 1, N2)"), ("nap", "NAP (vs the expert, N2)"))):
        ox = 60 + 450 * j
        b.append(f'<text class="lbl strong" x="{ox}" y="190">{title}</text>')
        for i, (d, f1) in enumerate(lb[key]):
            yy = 214 + 34 * i
            w = 300 * f1 / 0.8
            b.append(f'<rect class="bar-ink" x="{ox + 130}" y="{yy}" width="{w:.1f}" height="22" rx="4"/>')
            b.append(f'<text class="lbl-s" x="{ox + 122}" y="{yy + 16}" text-anchor="end">{esc(PRETTY.get(d, d))}</text>')
            b.append(f'<text class="val" x="{ox + 136 + w:.1f}" y="{yy + 16}">{f1:.2f}</text>')
    b.append('<text class="note" x="60" y="335">F1 = how well detections match the expert (1.0 = perfect). '
             'Top three per set; neither lead is statistically resolved.</text>')
    return svg(350, "".join(b), "The benchmark: datasets, cache, detectors, one referee, leaderboard")


def scene_teachers(cards):
    desc = {"a7": ("the balanced one", "four sigma features, fixed thresholds"),
            "luna_spindles": ("the cautious one", "wavelet envelope, high bar"),
            "wonambi_ucsd": ("the eager one", "fires most often"),
            "yasa_default": ("the strict one", "three tests at once")}
    b = []
    for k, t in enumerate(T):
        x = 20 + 236 * k
        c = cards[t]
        b.append(f'<rect class="card" x="{x}" y="12" width="220" height="250" rx="16"/>')
        b.append(f'<rect class="f-{CSS_ID[t]}" x="{x}" y="12" width="220" height="10" rx="5"/>')
        b.append(marker(MARK[t], x + 30, 56, 11, f"f-{CSS_ID[t]}"))
        b.append(f'<text class="h-card" x="{x + 52}" y="63">{NAME[t]}</text>')
        b.append(f'<text class="lbl-s" x="{x + 20}" y="94">{desc[t][0]}</text>')
        b.append(f'<text class="lbl-xs" x="{x + 20}" y="112">{desc[t][1]}</text>')
        rows = [("spindles / N2 minute", f"{c['n2']:.1f}"),
                ("finds at 9 dB (planted)", f"{100 * c['rec9']:.0f}%"),
                ("false alarms / quiet min", f"{c['fa']:.3f}"),
                ("recall, Dawid–Skene", f"{c['ds_recall']:.2f}")]
        for i, (lab, v) in enumerate(rows):
            yy = 146 + 28 * i
            b.append(f'<text class="lbl-xs" x="{x + 20}" y="{yy}">{lab}</text>'
                     f'<text class="val" x="{x + 200}" y="{yy}" text-anchor="end">{v}</text>')
            b.append(f'<line class="row" x1="{x + 20}" x2="{x + 200}" y1="{yy + 9}" y2="{yy + 9}"/>')
    return svg(276, "".join(b), "The four teachers and their measured habits")


def scene_clean(a):
    x0, x1 = 120, 930
    sx = lambda t: x0 + (x1 - x0) * t / a["dur"]
    lvl = {"W": 0, "REM": 1, "N1": 2, "N2": 3, "N3": 4}
    b = []
    for i, s in enumerate(a["stages"]):
        if s not in lvl:
            continue
        y = 40 + 22 * lvl[s]
        cls = "hyp-n2" if s == "N2" else "hyp"
        b.append(f'<rect class="{cls}" x="{sx(i * a["epoch"]):.2f}" y="{y}" width="{sx(a["epoch"]) - x0 + 0.4:.2f}" height="14"/>')
    for name, k in lvl.items():
        b.append(f'<text class="lbl-xs" x="{x0 - 10}" y="{51 + 22 * k}" text-anchor="end">{name}</text>')
    for s, e in a["flagged"]:
        b.append(f'<rect class="art" x="{sx(s):.2f}" y="30" width="{max(sx(e) - sx(s), 1.6):.2f}" height="128"/>')
    b.append(f'<text class="lbl" x="20" y="194">night {esc(a["rec"])} (CinC 2018): the dark lines are the {100 * a["share"]:.1f}% of 5-s windows flagged as artefact</text>')
    for h in range(0, int(a["dur"] // 3600) + 1):
        b.append(f'<text class="tick" x="{sx(h * 3600):.1f}" y="172" text-anchor="middle">{h} h</text>')
    for j, t in enumerate(("luna_spindles", "wonambi_ucsd")):
        bef, aft = a["counts"][t]
        y = 214 + 46 * j
        b.append(marker(MARK[t], 30, y + 10, 7, f"f-{CSS_ID[t]}"))
        b.append(f'<text class="lbl" x="44" y="{y + 15}">{NAME[t]}</text>')
        scale = 560 / max(aft, 1)
        b.append(f'<rect class="bar-ghost" x="{x0 + 40}" y="{y}" width="{max(bef * scale, 2):.1f}" height="9" rx="3"/>')
        b.append(f'<rect class="f-{CSS_ID[t]}" x="{x0 + 40}" y="{y + 12}" width="{aft * scale:.1f}" height="9" rx="3"/>')
        b.append(f'<text class="lbl-xs" x="{x0 + 46 + max(bef * scale, 2):.1f}" y="{y + 8}">{bef} spindles on the whole night</text>')
        b.append(f'<text class="lbl-xs" x="{x0 + 46 + aft * scale:.1f}" y="{y + 20}">{aft} on clean windows</text>')
    return svg(310, "".join(b), "A night with artefacts, the flagged windows, and teacher counts before and after cleaning")


def scene_patterns(t):
    rows = t.sort_values("share_of_valid_samples", ascending=False)
    show = rows.head(8)
    b = ['<text class="lbl-xs" x="40" y="26">which teachers said "spindle"</text>',
         '<text class="lbl-xs" x="330" y="26">share of N2 samples</text>',
         '<text class="lbl-xs" x="610" y="26">vote ≥ 2</text><text class="lbl-xs" x="720" y="26">Dawid–Skene P(spindle)</text>']
    for k, t_ in enumerate(T):
        b.append(f'<text class="lbl-xs" x="{52 + 50 * k}" y="44" text-anchor="middle">{NAME[t_]}</text>')
    for i, r in enumerate(show.itertuples()):
        y = 66 + 34 * i
        for k, t_ in enumerate(T):
            on = getattr(r, t_) == 1
            b.append(marker(MARK[t_], 52 + 50 * k, y, 8, f"f-{CSS_ID[t_]}" if on else "off"))
        share = r.share_of_valid_samples
        w = 190 * np.sqrt(share)
        b.append(f'<rect class="bar-ink" x="330" y="{y - 9}" width="{max(w, 2):.1f}" height="18" rx="3"/>')
        b.append(f'<text class="val" x="{336 + w:.1f}" y="{y + 5}">{100 * share:.1f}%</text>')
        b.append(f'<text class="val" x="626" y="{y + 5}" text-anchor="middle">{"yes" if r.mv else "no"}</text>')
        ds = r.ds
        b.append(f'<rect class="bar-ghost" x="720" y="{y - 7}" width="160" height="14" rx="7"/>')
        b.append(f'<rect class="bar-stu" x="720" y="{y - 7}" width="{max(160 * ds, 2):.1f}" height="14" rx="7"/>')
        b.append(f'<text class="val" x="930" y="{y + 5}" text-anchor="end">{ds:.2f}</text>')
    b.append(f'<text class="note" x="40" y="{66 + 34 * 8 + 10}">Bar length uses a square-root scale so the small patterns stay visible. '
             f'8 of the 16 patterns shown.</text>')
    return svg(66 + 34 * 8 + 24, "".join(b), "Teacher vote patterns, how common each is, and the targets they become")


def scene_student():
    b = []
    # training-only region
    b.append('<rect class="train-only" x="300" y="292" width="640" height="232" rx="18"/>')
    b.append('<text class="tag" x="318" y="316">TRAINING ONLY · thrown away afterwards</text>')
    # input
    wave = " ".join(f"{20 + i * 1.6:.1f},{150 + 14 * np.sin(i / 2.2) * (1.4 if 40 < i < 65 else 0.5):.1f}" for i in range(100))
    b.append(f'<polyline class="eeg" points="{wave}"/>')
    b.append('<text class="lbl strong" x="20" y="110">EEG in</text><text class="lbl-xs" x="20" y="190">40 s, 100 Hz</text>')
    # trunk U
    blocks = [(200, 60, "100 Hz"), (260, 120, "25 Hz"), (320, 180, "6.25 Hz")]
    for x, y, lab in blocks:
        b.append(f'<rect class="stu" x="{x}" y="{y}" width="56" height="44" rx="10"/><text class="lbl-in-xs" x="{x + 28}" y="{y + 27}" text-anchor="middle">{lab}</text>')
    for x, y, lab in [(400, 120, "25 Hz"), (460, 60, "100 Hz")]:
        b.append(f'<rect class="stu" x="{x}" y="{y}" width="56" height="44" rx="10"/><text class="lbl-in-xs" x="{x + 28}" y="{y + 27}" text-anchor="middle">{lab}</text>')
    b.append('<path class="arrow" d="M182,150 C190,120 190,90 198,84" marker-end="url(#ah)"/>')
    for (xa, ya), (xb, yb) in [((228, 104), (272, 120)), ((288, 164), (332, 180)), ((376, 202), (412, 164)), ((436, 120), (472, 104))]:
        b.append(f'<path class="arrow" d="M{xa},{ya} L{xb},{yb}" marker-end="url(#ah)"/>')
    b.append('<path class="skip" d="M256,82 L458,82"/><path class="skip" d="M316,142 L398,142"/>')
    b.append('<text class="lbl strong" x="200" y="36">The trunk: SUMOv2\'s U-Net, 70,338 parameters</text>')
    b.append('<text class="lbl-xs" x="190" y="262">squeezes 40 s down to 6.25 Hz for context,</text>'
             '<text class="lbl-xs" x="190" y="278">then back up to every 10 ms</text>')
    # heads
    b.append('<path class="arrow" d="M518,82 L590,82" marker-end="url(#ah)"/>')
    pw = " ".join(f"{600 + i * 2.2:.1f},{96 - 34 * (1 if 30 < i < 52 or 88 < i < 106 else 0.04):.1f}" for i in range(140))
    b.append(f'<polyline class="pline" points="{pw}"/>')
    b.append('<text class="lbl strong" x="600" y="36">p(t): chance of a spindle</text><text class="lbl-xs" x="600" y="122">the only output used later</text>')
    b.append('<path class="arrow" d="M518,96 C560,150 580,170 598,172" marker-end="url(#ah)"/>')
    bw = " ".join(f"{600 + i * 2.2:.1f},{200 - 20 * (np.exp(-((i - 30) ** 2) / 6) + np.exp(-((i - 52) ** 2) / 6) + np.exp(-((i - 88) ** 2) / 6) + np.exp(-((i - 106) ** 2) / 6)):.1f}" for i in range(140))
    b.append(f'<polyline class="bline" points="{bw}"/><text class="lbl-xs" x="600" y="224">b(t): where spindles start and end (optional)</text>')
    # stream B
    b.append('<rect class="card" x="320" y="336" width="250" height="160" rx="14"/>')
    b.append('<text class="lbl strong" x="336" y="362">Stream B: the teachers\' glasses</text>')
    for k, f in enumerate((7, 9.5, 10.5, 11.5, 12.5, 13.5, 15, 20)):
        cx = 340 + 27 * k
        hh = 32
        bump = " ".join(f"{cx - 11 + j:.1f},{430 - hh * np.exp(-((j - 11) ** 2) / 20):.1f}" for j in range(23))
        b.append(f'<polyline class="bump" points="{bump}"/><text class="lbl-xxs" x="{cx}" y="446" text-anchor="middle">{f:g}</text>')
    b.append('<text class="lbl-xs" x="336" y="470">8 learnable filters (Hz) → envelope → its level</text>'
             '<text class="lbl-xs" x="336" y="486">relative to the last few seconds · 56 parameters</text>')
    b.append('<path class="arrow" d="M120,170 C140,330 220,410 318,412" marker-end="url(#ah)"/>')
    # noise branch
    b.append('<rect class="card" x="610" y="336" width="314" height="160" rx="14"/>')
    b.append('<text class="lbl strong" x="626" y="362">The teacher profiler</text>')
    b.append('<text class="lbl-xs" x="626" y="380">uses the glasses and the trunk (dashed line)</text>')
    for k, t in enumerate(T):
        x = 630 + 72 * k
        b.append(f'<rect class="f-{CSS_ID[t]}" x="{x}" y="396" width="56" height="6" rx="3"/>')
        for i in range(2):
            for j in range(2):
                v = 0.85 if i == j else 0.15
                b.append(f'<rect class="cell" x="{x + 28 * j}" y="{408 + 28 * i}" width="26" height="26" rx="4" style="opacity:{0.25 + 0.75 * v:.2f}"/>')
                b.append(f'<text class="lbl-in-xxs" x="{x + 13 + 28 * j}" y="{425 + 28 * i}" text-anchor="middle">{v:.2f}</text>')
        b.append(f'<text class="lbl-xs" x="{x + 28}" y="482" text-anchor="middle">{NAME[t]}</text>')
    b.append('<path class="arrow" d="M572,416 L606,416" marker-end="url(#ah)"/>')
    b.append('<path class="arrow dashed" d="M378,206 C480,250 650,262 712,332" marker-end="url(#ah)"/>')
    b.append('<text class="note" x="626" y="512">starts at 0.85 on the diagonal: "right 85% of the time"</text>')
    return svg(540, "".join(b), "The student: trunk, outputs, and the training-only teacher glasses and teacher profiler")


def scene_learning(cards):
    p = 0.9
    y = cards["yasa_default"]
    q_yes = p * y["ds_recall"] + (1 - p) * (1 - y["ds_spec"])
    cost_plain = -np.log(1 - p)
    cost_noise = -np.log(1 - q_yes)
    steps = ["pick a 40-s piece of N2 sleep", "student guesses p(t)", "profiler says how each teacher errs",
             "compare with all 4 teachers' votes", "nudge the weights"]
    b = []
    cx, cy, R = 225, 205, 112
    b.append(f'<circle class="ring" cx="{cx}" cy="{cy}" r="{R}"/>')
    for k, s in enumerate(steps):
        ang = -np.pi / 2 + 2 * np.pi * k / len(steps)
        x, yy = cx + R * np.cos(ang), cy + R * np.sin(ang)
        b.append(f'<circle class="stop" cx="{x:.1f}" cy="{yy:.1f}" r="24"/><text class="lbl-in" x="{x:.1f}" y="{yy + 6:.1f}" text-anchor="middle">{k + 1}</text>')
        lx = x + (34 if np.cos(ang) >= -0.1 else -34)
        anchor = "start" if np.cos(ang) >= -0.1 else "end"
        words = s.split(" ")
        half = len(words) // 2 + 1
        for i, line in enumerate((" ".join(words[:half]), " ".join(words[half:]))):
            b.append(f'<text class="lbl-xs" x="{lx:.1f}" y="{yy - 4 + 15 * i:.1f}" text-anchor="{anchor}">{line}</text>')
    b.append(f'<text class="lbl strong" x="{cx}" y="{cy - 6}" text-anchor="middle">one batch</text>'
             f'<text class="lbl-xs" x="{cx}" y="{cy + 14}" text-anchor="middle">3,241 batches = one pass</text>'
             f'<text class="lbl-xs" x="{cx}" y="{cy + 30}" text-anchor="middle">then a check on 40 nights</text>')
    # the trick card
    x0 = 500
    b.append(f'<rect class="card" x="{x0}" y="20" width="440" height="378" rx="16"/>')
    b.append(f'<text class="lbl strong" x="{x0 + 22}" y="54">Why the profiler helps: one moment, one teacher</text>')
    lines = [f"The student is {100 * p:.0f}% sure there is a spindle. YASA stays silent.",
             "",
             "A plain student treats YASA's silence as a fact:",
             f"   surprise = −log(1 − {p:.1f}) = {cost_plain:.2f}",
             "so it learns to doubt itself.",
             "",
             f"The profiler knows YASA finds only about {100 * y['ds_recall']:.0f}% of spindles,",
             f"so silence was likely anyway: P(YASA says yes) = {q_yes:.2f}.",
             f"   surprise = −log(1 − {q_yes:.2f}) = {cost_noise:.2f}",
             "a much smaller push.",
             "",
             "Teacher habits are absorbed by the profiler; the student",
             "keeps what the teachers agree on. A trace penalty stops the",
             "profiler from explaining everything away."]
    for i, line in enumerate(lines):
        b.append(f'<text class="{"mono" if line.startswith("   ") else "lbl-s"}" x="{x0 + 22}" y="{88 + 22 * i}">{esc(line.strip())}</text>')
    b.append(f'<text class="note" x="{x0 + 22}" y="424">YASA\'s recall {y["ds_recall"]:.2f} is the label-free Dawid–Skene estimate on the corpus.</text>')
    return svg(440, "".join(b), "The training loop and a worked example of why modelling teacher errors helps")


def scene_recall(r):
    x0, x1, y0, y1 = 90, 600, 330, 40
    snrs = sorted(r.snr_db.unique())
    sx = lambda s: x0 + (x1 - x0) * (s - snrs[0]) / (snrs[-1] - snrs[0])
    sy = lambda v: y0 - (y0 - y1) * v
    b = [f'<rect class="zone" x="{sx(9):.1f}" y="{y1}" width="{x1 - sx(9) + 10:.1f}" height="{y0 - y1}" rx="6"/>']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(f'<line class="{"half" if v == 0.5 else "grid"}" x1="{x0}" x2="{x1 + 10}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                 f'<text class="tick" x="{x0 - 10}" y="{sy(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>')
    for s in snrs:
        b.append(f'<text class="tick" x="{sx(s):.1f}" y="{y0 + 22}" text-anchor="middle">{s:g} dB</text>')
    b.append(f'<text class="lbl-xs" x="{(x0 + x1) / 2}" y="{y0 + 46}" text-anchor="middle">strength of the planted spindle (SNR)</text>')
    b.append(f'<text class="lbl-xs" x="{sx(9) + 8:.1f}" y="{y1 + 18}">student can learn here</text>')
    b.append(f'<text class="lbl-xs" x="{x0 + 6}" y="{sy(0.5) - 8:.1f}">needed: half the teachers find it</text>')
    for t in T:
        g = r[r.teacher == t].sort_values("snr_db")
        pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in zip(g.snr_db, g.recall))
        b.append(f'<polyline class="s-{CSS_ID[t]}" points="{pts}"/>')
        for s, v in zip(g.snr_db, g.recall):
            b.append(marker(MARK[t], sx(s), sy(v), 4.5, f"f-{CSS_ID[t]}"))
    g = r[r.teacher == "mean of 4"].sort_values("snr_db")
    pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in zip(g.snr_db, g.recall))
    b.append(f'<polyline class="s-mean" points="{pts}"/>')
    # direct labels at the right end, nudged apart
    ends = [(t, float(r[(r.teacher == t) & (r.snr_db == snrs[-1])].recall.iloc[0])) for t in T + ["mean of 4"]]
    ends.sort(key=lambda e: -e[1])
    last = -1e9
    for t, v in ends:
        yy = max(sy(v) + 4, last + 15)
        last = yy
        name = NAME.get(t, "average of 4")
        b.append(f'<text class="{"lbl strong" if t == "mean of 4" else "lbl-xs"}" x="{x1 + 16}" y="{yy:.1f}">{name} {100 * v:.0f}%</text>')
    # right panel: none found
    nf = r[r.teacher == "none found"].sort_values("snr_db")
    b.append('<text class="lbl strong" x="730" y="52">missed by all four</text>')
    for i, (s, v) in enumerate(zip(nf.snr_db, nf.recall)):
        yy = 76 + 40 * i
        b.append(f'<text class="tick" x="770" y="{yy + 16}" text-anchor="end">{s:g} dB</text>')
        b.append(f'<rect class="bar-ghost" x="780" y="{yy}" width="150" height="22" rx="4"/>')
        b.append(f'<rect class="bar-ink" x="780" y="{yy}" width="{150 * v:.1f}" height="22" rx="4"/>')
        b.append(f'<text class="{"val-in" if v > 0.3 else "val"}" x="{784 + (150 * v if v <= 0.3 else 0):.1f}" y="{yy + 16}">{100 * v:.0f}%</text>')
    return svg(400, "".join(b), "Recall of planted spindles by strength for each teacher and their average")


def scene_exam(cons, lb):
    b = []
    b.append('<rect class="card" x="20" y="20" width="270" height="200" rx="16"/>')
    b.append('<text class="lbl strong" x="40" y="54">1 · choose, blind</text>')
    for i, line in enumerate(["checkpoint, threshold, decoder", "chosen on the 40 unlabelled", "validation nights only", "",
                              "written to frozen.json", "and committed"]):
        b.append(f'<text class="lbl-s" x="40" y="{84 + 20 * i}">{line}</text>')
    b.append('<path class="arrow" d="M296,120 L336,120" marker-end="url(#ah)"/>')
    # lock
    b.append('<rect class="card gold-card" x="342" y="20" width="270" height="200" rx="16"/>')
    b.append('<path class="lock" d="M452,92 v-18 a25,25 0 0 1 50,0 v18"/><rect class="lock-body" x="440" y="92" width="74" height="54" rx="10"/>')
    b.append('<text class="lbl strong" x="477" y="176" text-anchor="middle">2 · unlock DREAMS + NAP</text>'
             '<text class="lbl-s" x="477" y="198" text-anchor="middle">once; the code refuses before 1</text>')
    b.append('<path class="arrow" d="M618,120 L658,120" marker-end="url(#ah)"/>')
    b.append('<rect class="card" x="664" y="20" width="276" height="200" rx="16"/>')
    b.append('<text class="lbl strong" x="684" y="54">3 · same referee as Part A</text>')
    d, n = cons["dreams_spindles"], cons["nap"]
    best_d = max(d["teachers"].items(), key=lambda kv: kv[1])
    best_n = max(n["teachers"].items(), key=lambda kv: kv[1])
    rows = [("", "DREAMS", "NAP"),
            ("best teacher", f"{best_d[1]:.2f}", f"{best_n[1]:.2f}"),
            ("vote of 4 teachers", f"{d['mv']:.2f}", f"{n['mv']:.2f}"),
            ("SUMOv2 (expert-trained)", f"{dict(lb['dreams']).get('sumov2', float('nan')):.2f}", "—"),
            ("the student", "to come", "to come")]
    for i, (a_, b_, c_) in enumerate(rows):
        yy = 84 + 24 * i
        cls = "lbl-xs strong" if i == 0 or i == 4 else "lbl-xs"
        b.append(f'<text class="{cls}" x="684" y="{yy}">{a_}</text><text class="val" x="850" y="{yy}" text-anchor="end">{b_}</text>'
                 f'<text class="val" x="920" y="{yy}" text-anchor="end">{c_}</text>')
    b.append('<text class="note" x="20" y="250">F1 in N2 against the experts. The vote row needed no expert label either; the student has to beat it.</text>')
    return svg(266, "".join(b), "The final exam: choose blind, unlock the gold sets once, score with the benchmark's referee")


def scene_roadmap(pl):
    steps = [("1", "teachers and targets", "done"), ("2", "model, checks, pilot", "running" if pl and not pl["done"] else "done"),
             ("3", "CPU grid, 42 runs", "next"), ("4", "GPU grid, 140 runs", "later"),
             ("5", "choose blind, freeze", "later"), ("6", "one run on DREAMS + NAP", "later")]
    b = ['<line class="track" x1="60" x2="900" y1="70" y2="70"/>']
    for i, (n, lab, st) in enumerate(steps):
        x = 60 + 168 * i
        b.append(f'<circle class="node-{st}" cx="{x}" cy="70" r="20"/><text class="lbl-in" x="{x}" y="76" text-anchor="middle">{n}</text>')
        words = lab.split(" ")
        mid = (len(words) + 1) // 2
        for j, line in enumerate((" ".join(words[:mid]), " ".join(words[mid:]))):
            b.append(f'<text class="lbl-xs" x="{x}" y="{114 + 16 * j}" text-anchor="middle">{line}</text>')
        b.append(f'<text class="chip-{st}" x="{x}" y="34" text-anchor="middle">{st.upper()}</text>')
    return svg(160, "".join(b), "Roadmap of the six steps and their status")


# ------------------------------------------------------------------ page

CSS = """
:root{
  /* the page: a field guide; ink on cool paper, colour only for the four teachers; one scene per idea */
  --paper:#f6f7f9; --card:#ffffff; --ink:#14202e; --ink-2:#4a5566; --ink-3:#8a93a1; --line:#dde1e7;
  --a7:#2a78d6; --ucsd:#eb6834; --luna:#1baf7a; --yasa:#eda100; --hl:#e9edf3; --gold:#b8892a;
  --display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif; --body:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,Consolas,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#111821; --card:#18212c; --ink:#e8edf3; --ink-2:#aeb8c6; --ink-3:#7c8797; --line:#2b3644;
  --a7:#3987e5; --ucsd:#d95926; --luna:#199e70; --yasa:#c98500; --hl:#1f2a37; --gold:#d6a74a; color-scheme:dark;}}
:root[data-theme="dark"]{
  --paper:#111821; --card:#18212c; --ink:#e8edf3; --ink-2:#aeb8c6; --ink-3:#7c8797; --line:#2b3644;
  --a7:#3987e5; --ucsd:#d95926; --luna:#199e70; --yasa:#c98500; --hl:#1f2a37; --gold:#d6a74a; color-scheme:dark;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.55 var(--body)}
main{max-width:1040px;margin:0 auto;padding:40px 20px 80px;display:grid;gap:56px}
header{display:grid;gap:12px;padding-top:12px}
h1{font:700 44px/1.08 var(--display);margin:0;letter-spacing:-.01em;text-wrap:balance}
h2{font:700 27px/1.2 var(--display);margin:0;text-wrap:balance}
.lede{max-width:66ch;color:var(--ink-2);margin:0;font-size:19px}
.scene{display:grid;gap:14px}
.scene-head{display:flex;gap:14px;align-items:baseline}
.num{font:700 15px var(--mono);color:var(--ink-3);min-width:2.4em}
.scene p{margin:0;max-width:68ch;color:var(--ink-2)}
.code{font:13px var(--mono);color:var(--ink-3)}
.code b{color:var(--ink-2);font-weight:500}
.scene-fig{margin:0;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px;overflow-x:auto}
.scene-fig svg{display:block;width:100%;min-width:720px;height:auto}
.part{font:700 13px var(--mono);letter-spacing:.12em;color:var(--ink-3);text-transform:uppercase;border-top:1px solid var(--line);padding-top:18px}
.keys{display:flex;flex-wrap:wrap;gap:10px 22px;font-size:15px;color:var(--ink-2)}
.keys span{display:inline-flex;align-items:center;gap:8px}
.sw{width:14px;height:14px;border-radius:4px;display:inline-block}
.dash{width:22px;height:12px;border:2px dashed var(--ink-3);border-radius:4px;display:inline-block}
footer{color:var(--ink-3);font-size:14px;border-top:1px solid var(--line);padding-top:18px}
a{color:var(--a7)}
a:focus-visible{outline:2px solid var(--a7);outline-offset:2px}
/* svg roles */
svg text{font-family:var(--body);fill:var(--ink-2)}
.lbl{font-size:15px;fill:var(--ink)} .lbl-s{font-size:14px} .lbl-xs{font-size:12.5px} .lbl-xxs{font-size:10px;fill:var(--ink-3)}
.strong{font-weight:700;fill:var(--ink)} .h-card{font:700 24px var(--display);fill:var(--ink)}
.note{font-size:12.5px;fill:var(--ink-3);font-style:italic} .tick{font-size:11.5px;fill:var(--ink-3)}
.val{font:600 13px var(--mono);fill:var(--ink)} .val-in{font:600 13px var(--mono);fill:var(--paper)}
.mono{font:13px var(--mono);fill:var(--ink)} .tag{font:700 11.5px var(--mono);letter-spacing:.1em;fill:var(--ink-3)}
.lbl-in{font:700 15px var(--body);fill:var(--paper)} .lbl-in-s{font-size:13px;fill:var(--paper);opacity:.85}
.lbl-in-xs{font:600 11px var(--mono);fill:var(--paper)} .lbl-in-xxs{font:600 10px var(--mono);fill:var(--paper)}
.eeg{fill:none;stroke:var(--ink);stroke-width:1.2;stroke-linejoin:round}
.hl{fill:var(--hl)} .row{stroke:var(--line);stroke-width:1} .grid{stroke:var(--line)} .half{stroke:var(--ink-3);stroke-dasharray:5 4}
.f-a7{fill:var(--a7)} .f-ucsd{fill:var(--ucsd)} .f-luna{fill:var(--luna)} .f-yasa{fill:var(--yasa)} .f-ink{fill:var(--ink)}
.off{fill:none;stroke:var(--line);stroke-width:1.5}
.s-a7,.s-ucsd,.s-luna,.s-yasa{fill:none;stroke-width:2;stroke-linejoin:round}
.s-a7{stroke:var(--a7)} .s-ucsd{stroke:var(--ucsd)} .s-luna{stroke:var(--luna)} .s-yasa{stroke:var(--yasa)}
.s-mean{fill:none;stroke:var(--ink);stroke-width:4;stroke-linejoin:round}
.card{fill:var(--card);stroke:var(--line);stroke-width:1.5} .strong-card{stroke:var(--ink);stroke-width:2}
.gold-card{stroke:var(--gold);stroke-width:2}
.arrow{fill:none;stroke:var(--ink-3);stroke-width:2} .dashed{stroke-dasharray:6 5} .skip{fill:none;stroke:var(--ink-3);stroke-width:1.5;stroke-dasharray:3 4}
.bar-ink{fill:var(--ink)} .bar-ghost{fill:var(--hl)} .bar-stu{fill:var(--ink-2)}
.gold-fill{fill:var(--gold)} .free-fill{fill:var(--ink-2)} .leader{fill:none;stroke:var(--gold);stroke-width:2}
.hyp{fill:var(--ink-3);opacity:.55} .hyp-n2{fill:var(--ink-2)} .art{fill:var(--ink)}
.stu{fill:var(--ink)} .pline{fill:none;stroke:var(--ink);stroke-width:2.5} .bline{fill:none;stroke:var(--ink-3);stroke-width:2}
.bump{fill:none;stroke:var(--ink-2);stroke-width:1.8} .cell{fill:var(--ink)}
.train-only{fill:none;stroke:var(--ink-3);stroke-width:2;stroke-dasharray:8 6}
.stop{fill:var(--ink)} .ring{fill:none;stroke:var(--line);stroke-width:2}
.zone{fill:var(--hl)} .lock{fill:none;stroke:var(--gold);stroke-width:8;stroke-linecap:round} .lock-body{fill:var(--gold)}
.track{stroke:var(--line);stroke-width:4}
.node-done{fill:var(--ink)} .node-running{fill:var(--a7)} .node-next{fill:var(--ink-2)} .node-later{fill:var(--ink-3)}
.chip-done,.chip-running,.chip-next,.chip-later{font:700 11px var(--mono);letter-spacing:.08em}
.chip-done{fill:var(--ink)} .chip-running{fill:var(--a7)} .chip-next{fill:var(--ink-2)} .chip-later{fill:var(--ink-3)}
@media (max-width:520px){h1{font-size:34px} body{font-size:16px}}
"""


def scene(num, title, text, fig, code):
    return (f'<section class="scene"><div class="scene-head"><span class="num">{num}</span><h2>{esc(title)}</h2></div>'
            f'<p>{text}</p>{fig}<div class="code">{code}</div></section>')


def main():
    provenance.track(__file__, protocols=["benchmark.v1", "artefacts.v1"])
    c, h, lb, cons = crop_scene(), hours(), leaderboard(), consensus()
    art, cards, pat, rc = artefact_case(), teacher_cards(), patterns(), recall_curves()
    lo, hi = agreement()
    pl = pilot()
    m9 = float(rc[(rc.teacher == "mean of 4") & (rc.snr_db == 9.0)].recall.iloc[0])
    defs = ('<svg width="0" height="0" style="position:absolute"><defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            'style="fill:var(--ink-3)"/></marker></defs></svg>')
    keys = ('<div class="keys">' + "".join(
        f'<span><i class="sw" style="background:var(--{CSS_ID[t]})"></i>{NAME[t]} ({MARK[t]})</span>' for t in T) +
        '<span><i class="sw" style="background:var(--ink)"></i>the student</span>'
        '<span><i class="sw" style="background:var(--gold)"></i>expert-labelled (locked)</span>'
        '<span><i class="dash"></i>used in training only</span></div>')
    scenes = [
        '<div class="part">Part A · how good are today\'s detectors?</div>',
        scene("01", "What we are looking for",
              "A sleep spindle is a burst of 11–16 Hz waves lasting about half a second to two seconds, in light (N2) sleep. "
              "Below is a real 40-second stretch of one night. Under it, each row is one detector's answer, and the grey "
              "bands show where at least two of the four agree.",
              scene_spindle(c), f'<b>data:</b> HMC night SN001, {c["t0"]:.0f} s into the recording · '
              '<b>code:</b> detectors/*/run.py --clean, experiments/student/label_models.py'),
        scene("02", "Experts marked very little of it",
              "Drawing spindles by hand is slow, so only two small datasets have expert marks. Everything else has sleep "
              "stages but no spindles. That imbalance is the reason for Part B.",
              scene_hours(h), '<b>code:</b> spindlebench download · spindlebench cache → cache/manifest.csv'),
        scene("03", "One referee for every detector",
              "Every detector is wrapped so it reads the same files and writes the same table of events. One scoring code "
              "then compares each with the experts, using the same rules for all. Changing a rule changes the ranking, "
              "which is what the benchmark studies.",
              scene_benchmark(lb), '<b>code:</b> detectors/common.py · experiments/rq1_benchmark/evaluate.py · '
              'src/spindlebench/protocols/benchmark.py'),
        '<div class="part">Part B · can a detector learn without any expert?</div>',
        scene("04", "Four teachers with different habits",
              "Four rule-based detectors act as teachers. None of them ever saw an expert label. They agree on only "
              f"{lo:.0%}–{hi:.0%} of each other's events, and each has a measurable habit.",
              scene_teachers(cards), '<b>numbers:</b> results/student/teachers/density.csv · injection/teacher_recall.csv · '
              'work/targets/label_models.json'),
        scene("05", "First, clean the nights",
              "Movement and loose electrodes create huge bursts. Detectors that compare each moment with the whole night's "
              "level then set their bar too high and go almost silent. About 1% of each night is flagged and every "
              "teacher works on the rest.",
              scene_clean(art), '<b>code:</b> spindlebench artefacts (artefacts.v1) · detectors/common.py --clean'),
        scene("06", "Votes become lessons",
              "At every 10 ms of the 1,200 hours of unlabelled N2 sleep, the four teachers either fire or not: 16 possible "
              "patterns. Each pattern becomes a target for the student, here the simple vote and the Dawid–Skene estimate, "
              "which weighs teachers by how reliable they look.",
              scene_patterns(pat), '<b>code:</b> experiments/student/rasterise.py · label_models.py fit'),
        scene("07", "The student",
              "The part that stays is a small U-Net, exactly the network of SUMOv2 (a published, expert-trained detector), but "
              "trained from scratch on teacher votes. During training it gets two helpers: glasses that see what the teachers "
              "measure, and a profiler that learns where each teacher goes wrong.",
              scene_student(), '<b>code:</b> experiments/student/model.py · details: docs/student_architecture.md'),
        scene("08", "How it learns",
              "Training repeats a short loop thousands of times. The key step is 3: the student is not blamed for disagreeing "
              "with a teacher who is known to miss things.",
              scene_learning(cards), '<b>code:</b> experiments/student/train.py · losses.py · validate.py'
              + (f' · <b>pilot now:</b> pass {pl["passes"]}, agreement with the vote {pl["f1_mv"]:.2f} (best so far {pl["best"]:.2f})' if pl else "")),
        scene("09", "Testing without answers",
              "To check the teachers without experts, we plant artificial spindles of known strength into real nights and "
              f"count how many each teacher finds. The method needs the average to pass 50%. It does from 9 dB ({m9:.0%}); "
              "weaker spindles are invisible to all four, so the student cannot learn them.",
              scene_recall(rc), '<b>code:</b> experiments/student/build_val_injection.py · score_val_injection.py'),
        scene("10", "The final exam",
              "The expert-labelled sets stay locked until the student is chosen using unlabelled nights alone. Then it runs "
              "once and is scored by the same referee as every detector in Part A.",
              scene_exam(cons, lb), '<b>code:</b> data.guard() refuses DREAMS and NAP · selection.py → frozen.json · infer.py'),
        scene("11", "Where we are", "Steps 1 and 2 are done" + ("; the pilot student is training." if pl and not pl["done"] else ", including the pilot student."),
              scene_roadmap(pl), '<b>more:</b> ARCHITECTURE.md (precise diagrams) · experiments/student/README.md (commands)'),
    ]
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spindle student visual guide</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=JetBrains+Mono:wght@400;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body>{defs}<main>
<header><h1>Teaching a spindle detector without experts</h1>
<p class="lede">A picture guide to this project, from raw sleep EEG to a detector that never saw an expert label.
Every trace and number below comes from the project's own results.</p>{keys}</header>
{''.join(scenes)}
<footer>Generated by experiments/student/visual_guide.py from the stored results; regenerate it rather than editing it.
Teacher colours are paired with fixed shapes (A7 circle, Luna triangle, UCSD square, YASA diamond) so they can be told apart
without colour.</footer>
</main></body></html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(OUT, f"{len(page) / 1024:.0f} kB")


if __name__ == "__main__":
    main()
