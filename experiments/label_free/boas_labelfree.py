"""
Third dataset without expert events: do the detectors agree with each other, and are their
detections physiologically plausible, on 20 BOAS nights (healthy adults, PSG C3 against M1)?

    uv run python experiments/label_free/boas_labelfree.py

Detectors: four of the five shown in the slides (SUMOv2, SUMO; Mölle 2011, DREAMS-tuned YASA), stage-free,
C3. OpenSpindleNet is not run on BOAS yet (slowest on full nights).
BOAS has no spindle labels, so nothing here is an accuracy measure:
  agreement    each detector against a leave-one-out majority consensus of the others (an event that
               a majority of them detect, centres within ±0.5 s, merged), N2 only; plus pairwise F1
               between detectors. Shows whether a detector finds what the others find.
  plausibility N2 density (events/min; ~2-7 expected in adults), wake/N2 density ratio (should be
               near 0: the experts on NAP mark none in wake), and the median sigma peak frequency of
               the detected events (expected 11-16 Hz).
Writes results/boas/{agreement,pairwise,plausibility}.csv and plausibility.png.
    --max-sd 60   only nights with C3 SD < 60 µV (artefact screen) -> results/boas/clean/
"""
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from spindlebench.data.cache import signal_path
from spindlebench import paths, plotting
from spindlebench.data.cache import load_hypno
from spindlebench.metrics import events as em
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

DETS = ["sumov2", "sumo", "wonambi_moelle2011", "yasa_dreams_tuned"]  # OpenSpindleNet not run on BOAS yet
LABEL = {"sumov2": "SUMOv2", "sumo": "SUMO", "openspindlenet": "OpenSpindleNet",
         "wonambi_moelle2011": "Mölle 2011", "yasa_dreams_tuned": "YASA (DREAMS-tuned)"}
STORE = paths.RESULTS / "detections/boas"
OUT = paths.RESULTS / "boas"


def centres(ev):
    return ev[:, 0] + ev[:, 1] / 2


def consensus(evs, k, tol=0.5):
    """events (as centre points) found by at least k of the detector lists in evs"""
    pts = np.sort(np.concatenate([centres(e) for e in evs])) if evs else np.zeros(0)
    out, i = [], 0
    while i < len(pts):  # cluster centres within tol of the cluster's first point
        j = i
        while j + 1 < len(pts) and pts[j + 1] - pts[i] <= tol:
            j += 1
        c = pts[i:j + 1].mean()
        if sum(np.any(np.abs(centres(e) - c) <= tol + em.EPS) for e in evs) >= k:
            out.append(c)
        i = j + 1
    return np.c_[np.array(out), np.zeros(len(out))]


def peak_freq(x, sf, ev):
    """median over events of the 10-17 Hz Welch peak inside each event"""
    pk = []
    for o, d in ev:
        seg = x[int(o * sf):int((o + d) * sf)]
        if len(seg) < sf // 2:
            continue
        f, p = welch(seg - seg.mean(), sf, nperseg=len(seg), nfft=int(4 * sf))
        k = (f >= 10) & (f <= 17)
        pk.append(f[k][np.argmax(p[k])])
    return np.median(pk) if pk else np.nan


def main(max_sd=None):
    """max_sd: keep only nights whose C3 SD (µV) is below it (artefact screen); writes to results/boas/clean/"""
    global OUT
    m = pd.read_csv(paths.CACHE / "manifest.csv").query("dataset == 'boas'")
    if max_sd:
        sd = {r.rec: float(np.load(signal_path(r.path))["data"][list(np.load(signal_path(r.path))["ch_names"]).index("C3")].std()) for r in m.itertuples()}
        m = m[m.rec.map(sd) < max_sd]
        OUT = OUT / "clean"
    OUT.mkdir(parents=True, exist_ok=True)
    agree, pair, plaus = [], [], []
    for row in m.itertuples():
        is_n2 = load_hypno("boas", row.rec)
        h = pd.read_csv(paths.CACHE / "labels" / f"boas/{row.rec}__hypno.csv")
        minutes = h.stage.value_counts() * 0.5
        z = np.load(signal_path(row.path))
        x = z["data"][[str(c) for c in z["ch_names"]].index("C3")].astype(float)
        ev = {}
        for d in DETS:
            df = pd.read_csv(STORE / d / f"{row.rec}.csv")
            ev[d] = df[(df.channel.astype(str) == "C3") & (df.event == "spindle")][["onset", "duration"]].to_numpy(float)
        n2 = {d: em.in_mask(e, is_n2) for d, e in ev.items()}
        for d in DETS:
            others = [n2[o] for o in DETS if o != d]
            ref = consensus(others, k=len(others) // 2 + 1)
            tp, fp, fn, _ = em.match(np.c_[centres(n2[d]), np.zeros(len(n2[d]))], ref, "onset", 0.5)
            agree.append(dict(rec=row.rec, detector=d, tp=tp, fp=fp, fn=fn))
            st = h.stage.to_numpy()[np.clip((centres(ev[d]) // 30).astype(int), 0, len(h) - 1)] if len(ev[d]) else []
            dens = {s: (np.asarray(st) == s).sum() / minutes.get(s, np.nan) for s in ("W", "N2")}
            plaus.append(dict(rec=row.rec, detector=d, n2_per_min=dens["N2"], wake_per_min=dens["W"],
                              peak_hz=peak_freq(x, row.sf, n2[d])))
        for a, b in combinations(DETS, 2):
            tp, fp, fn, _ = em.match(np.c_[centres(n2[a]), np.zeros(len(n2[a]))],
                                     np.c_[centres(n2[b]), np.zeros(len(n2[b]))], "onset", 0.5)
            pair.append(dict(rec=row.rec, a=a, b=b, tp=tp, fp=fp, fn=fn))
        print(row.rec, flush=True)

    ag = pd.DataFrame(agree).groupby("detector")[["tp", "fp", "fn"]].sum()
    ag["F1_vs_others"] = 2 * ag.tp / (2 * ag.tp + ag.fp + ag.fn)
    ag["share_confirmed"] = ag.tp / (ag.tp + ag.fp)       # of its N2 events, found by a majority of the others
    ag["share_of_consensus"] = ag.tp / (ag.tp + ag.fn)    # of the others' consensus, found by it
    pl = pd.DataFrame(plaus)
    pls = pl.groupby("detector").agg(n2_per_min=("n2_per_min", "median"), wake_per_min=("wake_per_min", "median"),
                                     peak_hz=("peak_hz", "median"))
    pls["wake_to_n2"] = pls.wake_per_min / pls.n2_per_min
    res = ag.join(pls).reindex(DETS).reset_index()
    res.to_csv(OUT / "agreement.csv", index=False)
    pl.to_csv(OUT / "plausibility_per_night.csv", index=False)
    pw = pd.DataFrame(pair).groupby(["a", "b"])[["tp", "fp", "fn"]].sum()
    pw["F1"] = 2 * pw.tp / (2 * pw.tp + pw.fp + pw.fn)
    pw.reset_index().to_csv(OUT / "pairwise.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(pw.F1.round(3).unstack().to_string())

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, d in enumerate(DETS):
        g = pl[pl.detector == d]
        ax[0].scatter(np.full(len(g), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(g)), g.n2_per_min,
                      s=14, color=plotting.C_E1 if d in ("sumov2", "sumo") else plotting.C_YASA, alpha=0.6)
        ax[0].hlines(g.n2_per_min.median(), i - 0.3, i + 0.3, color=plotting.INK, lw=2)
    ax[0].axhspan(2, 7, color=plotting.GRID, alpha=0.6, lw=0)
    ax[0].set_xticks(range(len(DETS)), [LABEL[d] for d in DETS], rotation=25, ha="right", fontsize=8)
    ax[0].set_ylabel("spindles per minute of N2\n(dots = nights, bar = median; band 2-7)")
    ax[1].barh([LABEL[d] for d in DETS][::-1], res.F1_vs_others[::-1],
               color=[plotting.C_E1 if d in ("sumov2", "sumo") else plotting.C_YASA for d in DETS][::-1])
    ax[1].set_xlabel("F1 against the majority of the other detectors (N2)")
    ax[1].set_xlim(0, 1)
    for a in ax:
        plotting.style(a)
    fig.suptitle(f"BOAS, {len(m)} nights, C3 (no expert labels: agreement and plausibility only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "plausibility.png", dpi=170)
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    import sys
    main(float(sys.argv[sys.argv.index("--max-sd") + 1]) if "--max-sd" in sys.argv else None)
