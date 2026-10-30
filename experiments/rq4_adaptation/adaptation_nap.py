"""
E4 (RQ4) groundwork: which threshold-adaptation strategy recovers YASA performance on
NAP EEG? Uses the C3 threshold sweep written by experiments/yasa_sweeps/nap_pipeline.py
(results/nap/sweep_C3_center05.csv; centre ±0.5 s vs expert).

    uv run python experiments/rq4_adaptation/adaptation_nap.py

Every subject napped twice, so per-subject calibration is a clean experiment:
tune on one nap, test on the other. Strategies, cheapest first, all scored on the
same held-out recordings (included subjects with both naps included):
  default        published YASA thresholds
  transfer       thresholds tuned on DREAMS (results/dreams/sweep_expert1_onset.csv)
  population     tuned on all *other* NAP subjects (leave-one-subject-out)
  per_subject    tuned on the same subject's other nap only
  oracle         tuned on the test recording itself (upper bound, not a strategy)
(A "per-subject + population" rung, tuned on the pooled counts of the other nap and the
other subjects, was dropped: one nap's counts are swamped by the population's, so it chose
the population thresholds for every N2 nap and added no information.)
Scored on the 26 naps of the 13 included subjects with both naps included.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spindlebench import paths, plotting, yasa_utils
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

KEYS = ["rel_pow", "corr", "rms"]
OUT = paths.RESULTS / "adaptation"
OUT.mkdir(parents=True, exist_ok=True)


def f1(df):
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan


def best(df, fallback=None):
    """thresholds with the best pooled F1; fallback when df has no events at all
    (e.g. a nap without N2)"""
    g = df.groupby(KEYS)[["tp", "fp", "fn"]].sum()
    f = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
    return fallback if f.isna().all() else f.idxmax()


def at(df, th):
    return df[np.isclose(df[KEYS].to_numpy(float), np.array(th, float)).all(axis=1)]


sw = pd.read_csv(paths.RESULTS / "nap/sweep_C3_center05.csv")
ds = pd.read_csv(paths.RESULTS / "dreams/sweep_expert1_onset.csv").groupby(KEYS)[["tp", "fp", "fn"]].sum()
transfer = (2 * ds.tp / (2 * ds.tp + ds.fp + ds.fn)).idxmax()
default = (yasa_utils.YASA_DEFAULT["rel_pow"], yasa_utils.YASA_DEFAULT["corr"], yasa_utils.YASA_DEFAULT["rms"])

rows = []
for scope, sfx in (("all", ""), ("N2", "_N2")):
    s = sw.rename(columns={f"tp{sfx}": "TP", f"fp{sfx}": "FP", f"fn{sfx}": "FN"})
    s = s.drop(columns=[c for c in ["tp", "fp", "fn", "tp_N2", "fp_N2", "fn_N2"] if c in s.columns])
    s = s.rename(columns={"TP": "tp", "FP": "fp", "FN": "fn"})
    inc = s[~s.excluded]
    both = inc.groupby("subject").day.nunique()
    subjects = both[both == 2].index
    for subj in subjects:
        others = inc[inc.subject != subj]
        pop = best(others)
        for day in (1, 2):
            test = inc[(inc.subject == subj) & (inc.day == day)]
            calib = inc[(inc.subject == subj) & (inc.day != day)]
            choices = {"default": default, "transfer": transfer, "population": pop,
                       "per_subject": best(calib, pop), "oracle": best(test, pop)}
            for strat, th in choices.items():
                r = at(test, th)
                rows.append(dict(scope=scope, subject=subj, day=day, strategy=strat,
                                 rel_pow=th[0], corr=th[1], rms=th[2],
                                 tp=int(r.tp.sum()), fp=int(r.fp.sum()), fn=int(r.fn.sum())))

res = pd.DataFrame(rows)
res["f1"] = 2 * res.tp / (2 * res.tp + res.fp + res.fn)
res.to_csv(OUT / "nap_threshold_adaptation_per_recording.csv", index=False)
order = ["default", "transfer", "population", "per_subject", "oracle"]
summ = (res.groupby(["scope", "strategy"])
        .apply(lambda g: pd.Series({"n_rec": len(g), "F1_pooled": f1(g), "F1_mean": g.f1.mean(), "F1_sd": g.f1.std()}),
               include_groups=False)
        .reset_index())
summ["strategy"] = pd.Categorical(summ.strategy, order)
summ = summ.sort_values(["scope", "strategy"])
summ.to_csv(OUT / "nap_threshold_adaptation_summary.csv", index=False)
print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# paired comparison per recording vs population (N2 scope)
n2 = res[res.scope == "N2"].pivot_table(index=["subject", "day"], columns="strategy", values="f1")
for s in ["per_subject", "transfer"]:
    d = (n2[s] - n2["population"]).dropna()
    print(f"N2 {s:12s} - population: mean {d.mean():+.3f}, better on {(d > 0).sum()}/{len(d)} recordings")

fig, ax = plt.subplots(figsize=(8, 4))
sub = res[res.scope == "N2"]
for k, st in enumerate(order):
    v = sub[sub.strategy == st].f1.dropna().values
    ax.scatter(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, s=20, color=plotting.C_YASA, alpha=0.5,
               edgecolor="white", lw=0.5)
    ax.hlines(np.median(v), k - 0.28, k + 0.28, color=plotting.INK, lw=2)
    ax.text(k + 0.3, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=9)
ax.set_xticks(range(len(order)), ["defaults", "DREAMS\ntransfer", "population\n(LOSO)", "per-subject\n(other nap)",
                                  "oracle\n(upper bound)"])
ax.set_ylabel("F1 per recording (C3, N2, centre ±0.5 s)\nbar = median; 26 naps")
ax.set_ylim(0, 1)
ax.yaxis.grid(True, color=plotting.GRID, lw=0.8)
ax.set_axisbelow(True)
plotting.style(ax)
fig.tight_layout()
fig.savefig(OUT / "nap_threshold_adaptation.png", dpi=170)
print(f"wrote {OUT}/")
