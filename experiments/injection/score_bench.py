"""
Evaluation benchmark, step 3: score detectors on injected spindles (plan section 4).

    .venv/Scripts/python.exe injection/score_bench.py [--variants 0,1]

Matching: detection centre vs injected-event centre within +-0.5 s, one-to-one (Hungarian).
Only detections whose centre lies in the valid region (quiet N2 + injected events) are considered.
Quiet N2 = stretches where none of the benchmark detectors fired on the ORIGINAL signal, so a
detection there that matches no injected event is an "induced" false detection (caused by the
injection changing the detector's normalisation or splitting/shifting events). Precision is
therefore not measured here; it comes from the expert-label benchmark.

Outputs in results/injection/bench/:
  matches.csv      one row per injected event x detector (detected or not, SNR, freq, duration)
  recall.csv       recall per detector x dataset x SNR level, 95% CI from a cluster bootstrap over
                   people (2000 resamples; Wilson CI, which treats events as independent, kept for reference)
  summary.csv      per detector x dataset: recall, SNR50 (both with bootstrap CI), recall by freq /
                   duration bin, induced FD/min
  snr50_pairs.csv  paired bootstrap CI of the SNR50 difference for every detector pair
  rank_check.csv   Kendall tau of detector rankings: injected recall vs expert-label recall and F1
                   (tau CI: bootstrap of the injected side only)
  recall_snr.png
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau

from spindlebench import paths
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

BENCH = paths.RESULTS / "injection" / "bench"
DETECTORS = ["sumov2", "sumo", "a7", "moelle2011_faithful", "yasa_dreams_tuned", "yasa_default", "openspindlenet"]
NAMES = {"sumov2": "SUMOv2", "sumo": "SUMO", "a7": "A7", "moelle2011_faithful": "Mölle",
         "yasa_dreams_tuned": "YASA tuned", "yasa_default": "YASA default", "openspindlenet": "OpenSpindleNet"}
TOL = 0.5
# expert-label protocol of the main benchmark (results/benchmark/summary.csv)
MAIN = {"dreams_spindles": dict(reference="expert1", scope="N2", rule="onset", tol=0.5),
        "nap": dict(reference="expert", scope="N2", rule="center", tol=0.5)}


def match(det_c, ref_c):
    """bool array over ref: matched"""
    hit = np.zeros(len(ref_c), bool)
    if not len(det_c) or not len(ref_c):
        return hit, 0
    d = np.abs(det_c[:, None] - ref_c[None, :])
    ok = d <= TOL + 1e-9
    if not ok.any():
        return hit, 0
    r, c = linear_sum_assignment(np.where(ok, d, d[ok].sum() + 1))
    good = ok[r, c]
    hit[c[good]] = True
    return hit, int(good.sum())


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def snr50(levels, rec):
    """SNR where recall crosses 0.5 (linear interpolation); nan if never/always above"""
    for i in range(1, len(levels)):
        if rec[i - 1] < 0.5 <= rec[i]:
            return levels[i - 1] + (0.5 - rec[i - 1]) * (levels[i] - levels[i - 1]) / (rec[i] - rec[i - 1])
    return levels[0] if rec[0] >= 0.5 else np.nan


def bootstrap(m, B=2000, seed=0):
    """Cluster bootstrap over people (DREAMS excerpt = person; NAP: both naps of a person together;
    both injection variants of a recording stay together). The same resample is used for every
    detector, so detector comparisons are paired.
    -> {dataset: dict(levels, detectors, R (B x D x L), Rall (B x D))}"""
    rng = np.random.default_rng(seed)
    out = {}
    for ds, g in m.groupby("dataset"):
        subj = np.sort(g.subject.unique())
        dets = [d for d in DETECTORS if d in set(g.detector)]
        lev = np.sort(g.snr_db.unique())
        si = {s: i for i, s in enumerate(subj)}
        K = np.zeros((len(subj), len(dets), len(lev)))
        N = np.zeros_like(K)
        gg = g.groupby(["subject", "detector", "snr_db"]).detected.agg(["sum", "count"]).reset_index()
        for s, d, snr, k_, n_ in gg[["subject", "detector", "snr_db", "sum", "count"]].itertuples(index=False):
            if d in dets:
                i, j, k = si[s], dets.index(d), int(np.searchsorted(lev, snr))
                K[i, j, k], N[i, j, k] = k_, n_
        W = rng.multinomial(len(subj), np.full(len(subj), 1 / len(subj)), size=B).astype(float)
        Kb = np.einsum("bs,sdl->bdl", W, K)
        Nb = np.einsum("bs,sdl->bdl", W, N)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ds] = dict(levels=lev, detectors=dets, R=Kb / Nb, Rall=Kb.sum(2) / Nb.sum(2), n_subjects=len(subj))
    return out


def ci(x, axis=0):
    lo, hi = np.nanpercentile(x, [2.5, 97.5], axis=axis)
    return lo, hi


def score(variants):
    rows, fd = [], []
    for v in variants:
        vd = BENCH / f"v{v}"
        man = pd.read_csv(paths.CACHE_INJ / f"v{v}" / "manifest.csv")
        for r in man.itertuples():
            ev = pd.read_csv(vd / "events" / r.dataset / f"{r.rec}.csv")
            mz = np.load(vd / "masks" / r.dataset / f"{r.rec}.npz")
            n, sf = int(mz["n"]), float(mz["sf"])
            valid = np.unpackbits(mz["quiet"])[:n].astype(bool)
            for on, du in zip(ev.onset, ev.duration):
                valid[int(on * sf):int((on + du) * sf) + 1] = True
            quiet_min = np.unpackbits(mz["quiet"])[:n].sum() / sf / 60
            ref_c = (ev.onset + ev.duration / 2).to_numpy()
            for d in DETECTORS:
                f = vd / "detections" / r.dataset / d / f"{r.rec}.csv"
                if not f.exists():
                    continue
                det = pd.read_csv(f)
                det = det[(det.channel.astype(str) == r.default_channel) & (det.event == "spindle")]
                c = (det.onset + det.duration.fillna(0) / 2).to_numpy()
                c = c[valid[np.clip((c * sf).astype(int), 0, n - 1)]]
                hit, tp = match(c, ref_c)
                for i, e in enumerate(ev.itertuples()):
                    rows.append(dict(variant=v, dataset=r.dataset, rec=r.rec, subject=r.subject, detector=d,
                                     snr_db=e.snr_db,
                                     freq=e.freq, duration=e.duration, detected=bool(hit[i])))
                fd.append(dict(variant=v, dataset=r.dataset, rec=r.rec, detector=d, induced=len(c) - tp,
                               quiet_min=quiet_min))
    return pd.DataFrame(rows), pd.DataFrame(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="0,1")
    a = ap.parse_args()
    m, fd = score([int(v) for v in a.variants.split(",")])
    m.to_csv(BENCH / "matches.csv", index=False)

    bs = bootstrap(m)
    rec = []
    for (ds, d, s), g in m.groupby(["dataset", "detector", "snr_db"]):
        k, n = int(g.detected.sum()), len(g)
        wlo, whi = wilson(k, n)
        b = bs[ds]
        lo, hi = ci(b["R"][:, b["detectors"].index(d), int(np.searchsorted(b["levels"], s))])
        rec.append(dict(dataset=ds, detector=d, snr_db=s, n=n, recall=k / n, lo=lo, hi=hi,
                        wilson_lo=wlo, wilson_hi=whi))
    rec = pd.DataFrame(rec)
    rec.to_csv(BENCH / "recall.csv", index=False)

    m["freq_bin"] = pd.cut(m.freq, [11, 12.5, 14, 16], labels=["11-12.5", "12.5-14", "14-16"], include_lowest=True)
    m["dur_bin"] = pd.cut(m.duration, [0, 0.7, 1.0, 2.1], labels=["<0.7", "0.7-1.0", ">1.0"])
    summ = []
    for (ds, d), g in m.groupby(["dataset", "detector"]):
        rr = rec[(rec.dataset == ds) & (rec.detector == d)].sort_values("snr_db")
        f = fd[(fd.dataset == ds) & (fd.detector == d)]
        b = bs[ds]
        j = b["detectors"].index(d)
        s50 = np.array([snr50(b["levels"], r) for r in b["R"][:, j, :]])
        r_lo, r_hi = ci(b["Rall"][:, j])
        s_lo, s_hi = ci(s50)
        row = dict(dataset=ds, detector=d, n=len(g), n_subjects=b["n_subjects"], recall=g.detected.mean(),
                   recall_lo=r_lo, recall_hi=r_hi,
                   snr50=snr50(rr.snr_db.to_numpy(), rr.recall.to_numpy()), snr50_lo=s_lo, snr50_hi=s_hi,
                   snr50_undefined_frac=float(np.isnan(s50).mean()),
                   induced_per_quiet_min=f.induced.sum() / f.quiet_min.sum())
        mid = g[g.snr_db.between(0, 6)]   # mid-SNR events, where detectors differ
        for b, gb in mid.groupby("freq_bin", observed=True):
            row[f"recall_mid_f{b}"] = gb.detected.mean()
        for b, gb in mid.groupby("dur_bin", observed=True):
            row[f"recall_mid_d{b}"] = gb.detected.mean()
        summ.append(row)
    summ = pd.DataFrame(summ)

    # paired SNR50 differences between detectors (same bootstrap resamples)
    pair = []
    for ds, b in bs.items():
        s50 = np.array([[snr50(b["levels"], r) for r in b["R"][:, j, :]] for j in range(len(b["detectors"]))])
        for i, d1 in enumerate(b["detectors"]):
            for j, d2 in enumerate(b["detectors"]):
                if i < j:
                    diff = s50[i] - s50[j]
                    lo, hi = ci(diff)
                    pair.append(dict(dataset=ds, a=d1, b=d2, snr50_diff=np.nanmedian(diff), lo=lo, hi=hi,
                                     excludes_0=bool(lo > 0 or hi < 0)))
    pd.DataFrame(pair).round(3).to_csv(BENCH / "snr50_pairs.csv", index=False)

    # rank check against the expert-label benchmark
    main_s = pd.read_csv(paths.RESULTS / "benchmark" / "summary.csv")
    checks = []
    for ds, prot in MAIN.items():
        q = main_s[(main_s.dataset == ds) & (main_s.event == "spindle")]
        for k, val in prot.items():
            q = q[q[k] == val]
        q = q.set_index("detector")
        s = summ[summ.dataset == ds].set_index("detector")
        s["expert_recall"] = q.R_pooled.reindex(s.index)
        s["expert_f1"] = q.F1_pooled.reindex(s.index)
        summ.loc[summ.dataset == ds, "expert_recall"] = s.expert_recall.to_numpy()
        summ.loc[summ.dataset == ds, "expert_f1"] = s.expert_f1.to_numpy()
        ok = s.dropna(subset=["expert_recall"])
        b = bs[ds]
        idx = [b["detectors"].index(d) for d in ok.index]
        for target in ["expert_recall", "expert_f1"]:
            t, p = kendalltau(ok.recall, ok[target])
            # bootstrap of the injected side only (expert-label values held fixed)
            tb = np.array([kendalltau(r[idx], ok[target].to_numpy())[0] for r in b["Rall"]])
            lo, hi = ci(tb)
            checks.append(dict(dataset=ds, injected="recall", expert=target, n_detectors=len(ok), tau=t,
                               tau_lo=lo, tau_hi=hi, p=p))
    summ.round(4).to_csv(BENCH / "summary.csv", index=False)
    pd.DataFrame(checks).round(4).to_csv(BENCH / "rank_check.csv", index=False)
    print(summ.round(3).to_string())
    print(pd.DataFrame(checks).round(3).to_string())
    plot(rec)


def plot(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#1F4E79", "#2E75B6", "#E07B39", "#6A9F3F", "#8E5EA2", "#B8B8B8", "#C0504D"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for i, ds in enumerate(["dreams_spindles", "nap"]):
        a = ax[i]
        for d, col in zip(DETECTORS, colors):
            g = rec[(rec.dataset == ds) & (rec.detector == d)].sort_values("snr_db")
            if not len(g):
                continue
            a.plot(g.snr_db, g.recall, "-o", ms=4, lw=2, color=col, label=NAMES[d])
            a.fill_between(g.snr_db, g.lo, g.hi, color=col, alpha=0.12, lw=0)
        a.axhline(0.5, color="#999", lw=0.8, ls=":")
        a.set_title("DREAMS background" if i == 0 else "NAP background", fontsize=11)
        a.set_xlabel("Injected spindle SNR (dB)")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e5e5")
    ax[0].set_ylabel("Recall of injected spindles")
    ax[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(BENCH / "recall_snr.png", dpi=130)


if __name__ == "__main__":
    main()
