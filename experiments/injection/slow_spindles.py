"""
Slow-spindle hypothesis (plan_injection.md, benchmark findings): does SUMOv2 miss slow REAL spindles
on NAP, more than other detectors, and does that explain its low NAP score?

    .venv/Scripts/python.exe injection/slow_spindles.py

For each real expert spindle in N2 (NAP: mark +-0.5 s; DREAMS: expert-1 interval, 0-1000 s) we
measure its peak frequency and local sigma SNR on the raw signal (injection/realism.features), and
whether each detector found it: detection centre within +-0.5 s of the spindle centre, one-to-one
matching, detections restricted to N2 (the zero-shot detections in results/detections).

Tests (cluster bootstrap over people, 2000 resamples, paired across detectors):
  1. recall by frequency bin, standardised to the dataset's SNR distribution (SNR quartiles),
     so slow spindles being weaker does not masquerade as a frequency effect;
  2. slow deficit = recall(>= 13 Hz) - recall(< 12 Hz), per detector, and its difference
     between SUMOv2 and each other detector (difference in differences);
  3. counterfactual: SUMOv2's NAP recall with NAP spindles re-weighted to DREAMS' frequency mix.
Spindles whose peak falls below 10 Hz (alpha/theta dominates the window) are excluded from the
frequency analysis and counted.
-> results/injection/slow_spindles/{spindles.csv, recall_by_freq.csv, deficit.csv, counterfactual.csv, recall_by_freq.png}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.data.cache import signal_path
from spindlebench.synth.generator import sigma_band, stage_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realism import features  # noqa: E402
from score_bench import NAMES, match  # noqa: E402
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

OUT = paths.RESULTS / "injection" / "slow_spindles"
DETECTORS = ["sumov2", "sumo", "a7", "moelle2011_faithful", "yasa_dreams_tuned", "openspindlenet"]
FBINS = [10, 12, 13, 14, 17]
FLAB = ["10-12", "12-13", "13-14", "14-17"]
B = 2000


def collect():
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    m = m[m.dataset.isin(["dreams_spindles", "nap"]) & ~m.excluded.astype(bool)]
    rows = []
    for r in m.itertuples():
        z = np.load(signal_path(r.path))
        ch = [str(c) for c in z["ch_names"]]
        x = z["data"][ch.index(r.default_channel)].astype(float)
        sf = float(z["sf"])
        n = len(x)
        n2 = stage_mask(r.dataset, r.rec, n, sf)
        in_n2 = lambda t: n2[np.clip((np.asarray(t) * sf).astype(int), 0, n - 1)]  # noqa: E731
        sig = sigma_band(x, sf)
        if r.dataset == "nap":
            on = pd.read_csv(paths.CACHE / "labels" / "nap" / f"{r.rec}__expert__spindle.csv").onset.to_numpy(float)
            c, win = on, np.c_[on - 0.5, np.full(len(on), 1.0)]
        else:
            lab = pd.read_csv(paths.CACHE / "labels" / r.dataset / f"{r.rec}__expert1__spindle.csv")
            lab = lab[lab.onset + lab.duration <= 1000]
            c, win = (lab.onset + lab.duration / 2).to_numpy(), lab[["onset", "duration"]].to_numpy()
        keep = in_n2(c)
        c, win = c[keep], win[keep]
        if not len(c):
            continue
        hits = {}
        for d in DETECTORS:
            det = pd.read_csv(paths.RESULTS / "detections" / r.dataset / d / f"{r.rec}.csv")
            det = det[(det.channel.astype(str) == r.default_channel) & (det.event == "spindle")]
            dc = (det.onset + det.duration.fillna(0) / 2).to_numpy()
            dc = dc[(dc <= 1000) if r.dataset != "nap" else np.ones(len(dc), bool)]
            hits[d], _ = match(dc[in_n2(dc)], c)
        for i in range(len(c)):
            f = features(x, sig, sf, win[i, 0], win[i, 1])
            if f is None:
                continue
            rows.append(dict(dataset=r.dataset, rec=r.rec, subject=r.subject, centre=c[i], peak_freq=f["peak_freq"],
                             snr_db=f["snr_db"], amp_ptp=f["amp_ptp"], **{d: bool(hits[d][i]) for d in DETECTORS}))
    return pd.DataFrame(rows)


def std_recall(g, w_snr):
    """recall per detector x freq bin, directly standardised to snr-stratum weights w_snr"""
    out = np.full((len(DETECTORS), len(FLAB)), np.nan)
    for j, fb in enumerate(FLAB):
        gf = g[g.fbin == fb]
        acc, wsum = np.zeros(len(DETECTORS)), 0.0
        for q, w in w_snr.items():
            gq = gf[gf.sbin == q]
            if len(gq):
                acc += w * gq[DETECTORS].to_numpy(float).mean(0)
                wsum += w
        if wsum:
            out[:, j] = acc / wsum
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = collect()
    s.to_csv(OUT / "spindles.csv", index=False)
    print("spindles:", s.groupby("dataset").size().to_dict(), "peak < 10 Hz:",
          s[s.peak_freq < 10].groupby("dataset").size().to_dict())
    rng = np.random.default_rng(0)
    rec_rows, def_rows, cf_rows = [], [], []
    fq = {}
    for ds, g in s.groupby("dataset"):
        g = g[g.peak_freq >= 10].copy()
        g["fbin"] = pd.cut(g.peak_freq, FBINS, labels=FLAB, right=False).astype(str)
        g["sbin"] = pd.qcut(g.snr_db, 4, labels=False)
        w_snr = g.sbin.value_counts(normalize=True).to_dict()
        fq[ds] = g.fbin.value_counts(normalize=True).reindex(FLAB).to_numpy()
        subj = g.subject.unique()
        groups = {u: g[g.subject == u] for u in subj}
        point = std_recall(g, w_snr)
        boot = []
        for _ in range(B):
            pick = rng.choice(subj, len(subj), replace=True)
            gb = pd.concat([groups[u] for u in pick])
            boot.append(std_recall(gb, w_snr))
        boot = np.array(boot)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        n_bin = g.fbin.value_counts().reindex(FLAB).to_numpy()
        for i, d in enumerate(DETECTORS):
            for j, fb in enumerate(FLAB):
                rec_rows.append(dict(dataset=ds, detector=d, freq_bin=fb, n=int(n_bin[j]), recall_std=point[i, j],
                                     lo=lo[i, j], hi=hi[i, j], recall_raw=g[g.fbin == fb][d].mean()))
        # slow deficit: fast (>= 13 Hz) minus slow (< 12 Hz), fast = mean of the two upper bins weighted by n
        wf = n_bin[2:] / n_bin[2:].sum()
        deficit = lambda R: (R[..., 2:] * wf).sum(-1) - R[..., 0]  # noqa: E731
        dp, db = deficit(point), deficit(boot)
        j0 = DETECTORS.index("sumov2")
        for i, d in enumerate(DETECTORS):
            l, h = np.nanpercentile(db[:, i], [2.5, 97.5])
            row = dict(dataset=ds, detector=d, deficit=dp[i], lo=l, hi=h)
            if d != "sumov2":
                dd = db[:, j0] - db[:, i]
                l2, h2 = np.nanpercentile(dd, [2.5, 97.5])
                row.update(sumov2_minus_this=dp[j0] - dp[i], did_lo=l2, did_hi=h2)
            def_rows.append(row)
    # counterfactual: NAP spindles re-weighted to DREAMS' frequency mix
    g = s[(s.dataset == "nap") & (s.peak_freq >= 10)].copy()
    g["fbin"] = pd.cut(g.peak_freq, FBINS, labels=FLAB, right=False).astype(str)
    g = g[g.fbin.isin(FLAB)]  # peaks >= 17 Hz have no bin
    w = dict(zip(FLAB, fq["dreams_spindles"] / fq["nap"]))
    g["w"] = g.fbin.map(w)
    subj = g.subject.unique()
    groups = {u: g[g.subject == u] for u in subj}
    for d in DETECTORS:
        raw, rew = g[d].mean(), np.average(g[d], weights=g.w)
        bs = []
        for _ in range(B):
            gb = pd.concat([groups[u] for u in rng.choice(subj, len(subj), replace=True)])
            bs.append(np.average(gb[d], weights=gb.w) - gb[d].mean())
        l, h = np.percentile(bs, [2.5, 97.5])
        cf_rows.append(dict(detector=d, nap_recall=raw, nap_recall_dreams_freq_mix=rew, change=rew - raw, lo=l, hi=h,
                            dreams_recall=s[(s.dataset == "dreams_spindles") & (s.peak_freq >= 10)][d].mean()))
    rec = pd.DataFrame(rec_rows)
    dfc = pd.DataFrame(def_rows)
    cf = pd.DataFrame(cf_rows)
    rec.round(4).to_csv(OUT / "recall_by_freq.csv", index=False)
    dfc.round(4).to_csv(OUT / "deficit.csv", index=False)
    cf.round(4).to_csv(OUT / "counterfactual.csv", index=False)
    print("frequency mix (10-12, 12-13, 13-14, 14-17):", {k: np.round(v, 2).tolist() for k, v in fq.items()})
    print(rec.pivot_table(index=["dataset", "detector"], columns="freq_bin", values="recall_std").round(2).to_string())
    print(dfc.round(3).to_string())
    print(cf.round(3).to_string())
    plot(rec)


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sumov2": "#1F4E79", "sumo": "#2E75B6", "a7": "#E07B39", "moelle2011_faithful": "#6A9F3F",
              "yasa_dreams_tuned": "#8E5EA2", "openspindlenet": "#C0504D"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    x = np.arange(len(FLAB))
    for k, ds in enumerate(["dreams_spindles", "nap"]):
        a = ax[k]
        for i, d in enumerate(DETECTORS):
            g = rec[(rec.dataset == ds) & (rec.detector == d)].set_index("freq_bin").reindex(FLAB)
            off = (i - 2.5) * 0.06
            lw = 2.8 if d == "sumov2" else 1.6
            a.errorbar(x + off, g.recall_std, yerr=[g.recall_std - g.lo, g.hi - g.recall_std], fmt="-o", ms=4,
                       lw=lw, capsize=0, color=colors[d], label=NAMES[d])
        n = rec[(rec.dataset == ds) & (rec.detector == "sumov2")].set_index("freq_bin").reindex(FLAB).n
        a.set_xticks(x, [f"{f} Hz\n(n={int(v)})" for f, v in zip(FLAB, n)])
        a.set_title("DREAMS (expert 1)" if k == 0 else "NAP (expert marks)", fontsize=11)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e5e5")
    ax[0].set_ylabel("Recall of real spindles\n(standardised to SNR mix)")
    ax[1].legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "recall_by_freq.png", dpi=130)


if __name__ == "__main__":
    main()
