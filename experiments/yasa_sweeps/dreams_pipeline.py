"""
YASA spindle benchmark on the DREAMS Sleep Spindles database (all 8 excerpts).

    uv run python experiments/yasa_sweeps/dreams_pipeline.py [DatabaseSpindles] [results/dreams] [--no-sweep]

Protocol
  channel     the scored channel: C3-A1 (excerpts 1, 3), Cz-A1 (others)
  references  expert 1, expert 2, union, intersection (onsets within 1 s)
  windows     expert 1 only scored the first 1000 s of every excerpt, so any
              reference involving expert 1 is scored on [0, 1000) s; expert 2
              alone is scored on the whole 1800 s. Expert 2 exists for 1-6 only.
  scope       "all" = whole window; "N2" = events whose midpoint falls in an
              N2 epoch of the expert hypnogram (5-s epochs, R&K code 2)
  matching    one-to-one; onset within 0.5 s, or IoU >= 0.2
  detectors   YASA published defaults, YASA "loosened" (Update 1), YASA tuned
              by leave-one-excerpt-out grid search (vs expert 1, onset, all),
              DREAMS' own automatic detector (Devuyst et al.)
  ceiling     expert 2 scored against expert 1 (inter-rater agreement)

DREAMS hypnogram codes are Rechtschaffen & Kales: 5 wake, 4 REM, 3 S1, 2 S2,
1 S3, 0 S4 (checked against band power: code 0 has the most delta).
"""
import itertools
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from spindlebench import paths, plotting, yasa_utils
from spindlebench.metrics import events as em
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

mne.set_log_level("ERROR")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ROOT = Path(args[0]) if args else paths.raw("dreams_spindles")
OUT = Path(args[1]) if len(args) > 1 else (paths.RESULTS / "dreams")
SWEEP = "--no-sweep" not in sys.argv
OUT.mkdir(parents=True, exist_ok=True)

E1_END = 1000.0
EPOCH = 5.0
STAGE_NAMES = {5: "W", 4: "REM", 3: "N1", 2: "N2", 1: "N3", 0: "N3"}  # S3+S4 -> N3
GRID = yasa_utils.THRESH_GRID


def read_events(path):
    if not path.exists():
        return None
    return np.loadtxt(path, skiprows=1, ndmin=2)


def load(n):
    raw = mne.io.read_raw_edf(ROOT / f"excerpt{n}.edf", preload=True)
    sf = raw.info["sfreq"]
    ch = next(c for c in raw.ch_names if c.upper().startswith(("C3", "CZ")))
    sig = raw.get_data(picks=ch)[0]
    if np.abs(sig).max() < 1e-2:  # volts -> µV; some DREAMS EDF headers already give µV
        sig = sig * 1e6
    hyp = np.loadtxt(ROOT / f"Hypnogram_excerpt{n}.txt", skiprows=1).astype(int)
    return dict(n=n, sf=sf, ch=ch, sig=sig, hyp=hyp, dur=len(sig) / sf,
                e1=read_events(ROOT / f"Visual_scoring1_excerpt{n}.txt"),
                e2=read_events(ROOT / f"Visual_scoring2_excerpt{n}.txt"),
                auto=read_events(ROOT / f"Automatic_detection_excerpt{n}.txt"))


def n2_mask(hyp):
    def f(t):
        idx = np.clip((t // EPOCH).astype(int), 0, len(hyp) - 1)
        return hyp[idx] == 2
    return f


def references(x):
    """name -> (events, window end)"""
    refs = {"expert1": (x["e1"], E1_END)}
    if x["e2"] is not None:
        e1w, e2w = em.clip(x["e1"], 0, E1_END), em.clip(x["e2"], 0, E1_END)
        refs["expert2"] = (x["e2"], x["dur"])
        refs["union"] = (em.union(e1w, e2w), E1_END)
        refs["intersection"] = (em.intersection(e1w, e2w), E1_END)
    return refs


def score_rows(det, x, detector, refs=None):
    rows = []
    for ref_name, (ref, t_end) in (refs or references(x)).items():
        for scope in ("all", "N2"):
            d, r = em.clip(det, 0, t_end), em.clip(ref, 0, t_end)
            if scope == "N2":
                d, r = em.in_mask(d, n2_mask(x["hyp"])), em.in_mask(r, n2_mask(x["hyp"]))
            for rule in em.RULES:
                tp, fp, fn, _ = em.match(d, r, rule)
                p, rc, f = em.prf(tp, fp, fn)
                rows.append(dict(excerpt=x["n"], detector=detector, reference=ref_name,
                                 scope=scope, rule=rule, tp=tp, fp=fp, fn=fn,
                                 precision=p, recall=rc, f1=f))
    return rows


# ---- load + describe ------------------------------------------------------
data = [load(n) for n in range(1, 9)]
desc = []
for x in data:
    stages = pd.Series([STAGE_NAMES.get(h, "?") for h in x["hyp"]]).value_counts(normalize=True)
    desc.append(dict(excerpt=x["n"], sf=x["sf"], channel=x["ch"], minutes=x["dur"] / 60,
                     n_expert1=len(x["e1"]), n_expert2=len(x["e2"]) if x["e2"] is not None else None,
                     n_expert1_N2=len(em.in_mask(em.clip(x["e1"], 0, E1_END), n2_mask(x["hyp"]))),
                     n_auto=len(x["auto"]),
                     **{f"pct_{s}": round(100 * stages.get(s, 0), 1) for s in ["W", "N1", "N2", "N3", "REM"]}))
desc = pd.DataFrame(desc)
desc.to_csv(OUT / "dataset_description.csv", index=False)
print(desc.to_string(index=False), "\n")

# ---- fixed detectors + inter-rater ceiling --------------------------------
rows, dets = [], {}
for x in data:
    for name, th in [("yasa_default", yasa_utils.YASA_DEFAULT), ("yasa_loose", yasa_utils.YASA_LOOSE)]:
        dets[(name, x["n"])] = yasa_utils.run_yasa(x["sig"], x["sf"], th)
        rows += score_rows(dets[(name, x["n"])], x, name)
    rows += score_rows(x["auto"], x, "dreams_auto")
    if x["e2"] is not None:  # expert 2 as a "detector" against expert 1
        rows += score_rows(x["e2"], x, "expert2_vs_expert1", {"expert1": (x["e1"], E1_END)})

# ---- threshold sweep + leave-one-excerpt-out tuning -----------------------
if SWEEP:
    combos = list(itertools.product(*GRID.values()))
    print(f"sweep: {len(combos)} threshold sets x {len(data)} excerpts")
    t0, sweep = time.time(), []
    for x in data:
        e1w = em.clip(x["e1"], 0, E1_END)
        for rp, co, rm in combos:
            det = yasa_utils.run_yasa(x["sig"], x["sf"], {"rel_pow": rp, "corr": co, "rms": rm})
            dets[((rp, co, rm), x["n"])] = det
            tp, fp, fn, _ = em.match(em.clip(det, 0, E1_END), e1w, "onset")
            sweep.append(dict(excerpt=x["n"], rel_pow=rp, corr=co, rms=rm, tp=tp, fp=fp, fn=fn))
        print(f"  excerpt {x['n']} done ({time.time() - t0:.0f} s)", flush=True)
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "sweep_expert1_onset.csv", index=False)

    keys = ["rel_pow", "corr", "rms"]
    pooled = sweep.groupby(keys)[["tp", "fp", "fn"]].sum()
    pooled["f1"] = 2 * pooled.tp / (2 * pooled.tp + pooled.fp + pooled.fn)
    best_all = pooled.f1.idxmax()
    loo = []
    for x in data:
        tr = sweep[sweep.excerpt != x["n"]].groupby(keys)[["tp", "fp", "fn"]].sum()
        best = (2 * tr.tp / (2 * tr.tp + tr.fp + tr.fn)).idxmax()
        loo.append(dict(held_out=x["n"], rel_pow=best[0], corr=best[1], rms=best[2]))
        rows += score_rows(dets[(best, x["n"])], x, "yasa_tuned_loo")
    pd.DataFrame(loo).to_csv(OUT / "loo_chosen_thresholds.csv", index=False)
    print(f"\nbest thresholds on all 8 (in-sample, optimistic): {dict(zip(keys, best_all))}, "
          f"pooled F1 {pooled.f1.max():.3f}")
    print("LOO choices:\n", pd.DataFrame(loo).to_string(index=False))

per = pd.DataFrame(rows)
per.to_csv(OUT / "per_excerpt.csv", index=False)
summ = em.summarise(per, ["detector", "reference", "scope", "rule"])
summ.to_csv(OUT / "summary.csv", index=False)
pd.set_option("display.width", 200, "display.precision", 3)
print("\n=== vs expert 1, onset rule ===")
print(summ.query("reference == 'expert1' and rule == 'onset'").to_string(index=False))
print("\n=== all references, onset rule, N2 ===")
print(summ.query("rule == 'onset' and scope == 'N2'").to_string(index=False))

# ---- figures --------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.edgecolor": plotting.INK2})

# 1. F1 per excerpt vs expert 1 (onset, all)
sub = per.query("reference == 'expert1' and rule == 'onset' and scope == 'all'")
order = [d for d in ["yasa_default", "yasa_loose", "yasa_tuned_loo", "dreams_auto", "expert2_vs_expert1"]
         if d in sub.detector.unique()]
labels = {"yasa_default": "YASA defaults", "yasa_loose": "YASA loosened", "yasa_tuned_loo": "YASA tuned (LOO)",
          "dreams_auto": "DREAMS automatic", "expert2_vs_expert1": "Expert 2 (human ceiling)"}
colors = {"yasa_default": "#9cc3ee", "yasa_loose": "#5b9be0", "yasa_tuned_loo": plotting.C_YASA,
          "dreams_auto": "#a7a6a0", "expert2_vs_expert1": plotting.C_E2}
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.8 / len(order)
for k, d in enumerate(order):
    s = sub[sub.detector == d].set_index("excerpt").f1.reindex(range(1, 9))
    ax.bar(np.arange(1, 9) + (k - (len(order) - 1) / 2) * w, s.values, w * 0.9,
           color=colors[d], label=labels[d], edgecolor="white", linewidth=1)
ax.set_xticks(range(1, 9), [f"excerpt {i}\n{int(desc.sf[i-1])} Hz" for i in range(1, 9)])
ax.set_ylabel("F1 vs expert 1 (onset ±0.5 s)")
ax.set_ylim(0, 1)
ax.yaxis.grid(True, color=plotting.GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=len(order), loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)
plotting.style(ax)
fig.tight_layout()
fig.savefig(OUT / "f1_per_excerpt.png", dpi=170)
plt.close(fig)

# 2. threshold sensitivity: pooled F1 over rel_pow x corr, one panel per rms
if SWEEP:
    rms_vals = GRID["rms"]
    fig, axes = plt.subplots(1, len(rms_vals), figsize=(3.0 * len(rms_vals), 3.3), sharey=True)
    pf = pooled.reset_index()
    for ax, rm in zip(axes, rms_vals):
        m = pf[pf.rms == rm].pivot(index="corr", columns="rel_pow", values="f1")
        im = ax.imshow(m.values, origin="lower", cmap="Blues", vmin=0, vmax=max(0.8, pf.f1.max()), aspect="auto")
        ax.set_xticks(range(len(m.columns)), [f"{v:.2f}" for v in m.columns], fontsize=8)
        ax.set_yticks(range(len(m.index)), [f"{v:.2f}" for v in m.index], fontsize=8)
        ax.set_title(f"rms = {rm}", fontsize=10)
        ax.set_xlabel("rel_pow")
        for (i, j), v in np.ndenumerate(m.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if v > 0.5 else plotting.INK)
        for name, th in [("D", yasa_utils.YASA_DEFAULT), ("L", yasa_utils.YASA_LOOSE)]:
            if th["rms"] == rm:
                ax.add_patch(plt.Rectangle((list(m.columns).index(th["rel_pow"]) - .5, list(m.index).index(th["corr"]) - .5),
                                           1, 1, fill=False, ec=plotting.C_E1, lw=2))
    axes[0].set_ylabel("corr")
    fig.suptitle("YASA pooled F1 vs expert 1 (8 excerpts, onset ±0.5 s). Orange boxes: defaults, loosened", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sensitivity.png", dpi=170)
    plt.close(fig)

# 3. timeline per excerpt: hypnogram + events (first 1000 s, where both experts scored)
fig, axes = plt.subplots(8, 1, figsize=(13, 13), sharex=True)
stage_y = {5: 4, 4: 3, 3: 2, 2: 1, 1: 0, 0: 0}
for ax, x in zip(axes, data):
    t = np.arange(len(x["hyp"])) * EPOCH
    ax.step(t, [stage_y[h] * 0.25 for h in x["hyp"]], where="post", color=plotting.INK2, lw=0.8)
    lanes = [("expert 1", x["e1"], plotting.C_E1, 1.35), ("expert 2", x["e2"], plotting.C_E2, 1.65),
             ("YASA defaults", dets[("yasa_default", x["n"])], "#9cc3ee", 1.95),
             ("YASA loosened", dets[("yasa_loose", x["n"])], plotting.C_YASA, 2.25)]
    for name, ev, c, y in lanes:
        if ev is not None and len(ev):
            ax.broken_barh([(o, max(d, 2)) for o, d in ev], (y - 0.1, 0.2), color=c, label=name)
    ax.axvline(E1_END, color=plotting.INK2, ls="--", lw=0.8)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["N3", "N2", "N1", "REM", "W"], fontsize=7)
    ax.set_ylim(-0.15, 2.45)
    ax.set_ylabel(f"ex {x['n']}", rotation=0, ha="right", va="center")
    plotting.style(ax)
axes[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.6), fontsize=9)
axes[-1].set_xlabel("time (s); dashed line = end of expert 1 scoring (1000 s)")
fig.tight_layout()
fig.savefig(OUT / "timelines.png", dpi=150)
plt.close(fig)

print(f"\nwrote {OUT}/")
