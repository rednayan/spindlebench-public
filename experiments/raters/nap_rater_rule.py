"""
NAP check 2 (reports/Spindle detector failure on NAP.md): score detectors under the NAP rater's own
marking rule. Mei et al. (2017, 2018): "Only those spindles that occurred across several channels
(n > 3) with minimal temporal variation (< 1 s) were marked", viewing F3, F4, C3, C4, O1, O2.

    uv run python experiments/raters/nap_rater_rule.py

For each detector run on all six channels, detections are clustered across channels: sorted by
centre, a cluster starts at the earliest unused detection and takes every later detection whose
centre is < 1 s after it; a cluster with detections on >= 4 distinct channels becomes one
multichannel event (median onset, median offset). Those events are scored exactly like the C3
benchmark (experiments/nap_scoring/nap_scoring.py: N2 scope, tolerance sweep + integrated F1, affiliation, containment).
Detections: results/detections/nap (6-channel runs from the channel ablation) and, for A7 and Mölle
2011 (faithful), results/multichannel/detections/nap (run 2026-10-28 with --channels C3,F3,F4,C4,O1,O2).
-> results/nap_scoring/rater_rule.csv
"""

import numpy as np
import pandas as pd

from spindlebench import paths, registry
from spindlebench.data import nap
from spindlebench.protocols import nap_points as npt
from spindlebench import provenance  # noqa: E402
from spindlebench.protocols import nap_points  # noqa: E402

provenance.track(__file__, protocols=[nap_points.VERSION])

ROOT = paths.ROOT
CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2"]
MIN_CH, WIN = 4, 1.0
STORES = [paths.RESULTS / "multichannel" / "detections" / "nap", paths.RESULTS / "detections" / "nap"]
B = 2000


def load_all(det, rec):
    for s in STORES:
        f = s / det / f"{rec}.csv"
        if f.exists():
            d = pd.read_csv(f)
            return d[(d.event == "spindle") & d.channel.isin(CHANNELS)]
    return None


def cluster(d):
    if not len(d):
        return np.zeros(0), np.zeros(0)
    on = d.onset.to_numpy(float)
    off = on + d.duration.fillna(0).to_numpy(float)
    c = (on + off) / 2
    ch = d.channel.to_numpy()
    o = np.argsort(c)
    on, off, c, ch = on[o], off[o], c[o], ch[o]
    used = np.zeros(len(c), bool)
    ev_on, ev_off = [], []
    for i in range(len(c)):
        if used[i]:
            continue
        j = np.flatnonzero(~used & (c >= c[i]) & (c - c[i] < WIN))
        if len(set(ch[j])) >= MIN_CH:
            used[j] = True
            ev_on.append(np.median(on[j]))
            ev_off.append(np.median(off[j]))
    return np.array(ev_on), np.array(ev_off)


def main():
    infos = [nap.nap_info(r) for r in nap.naps().itertuples()]
    subjects = sorted({i["subject"] for i in infos})
    cands = sorted({p.name for s in STORES if s.exists() for p in s.iterdir()} - registry.NAP_EXCLUDE)
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=B).astype(float)
    rows = []
    for det in cands:
        per = [(i, load_all(det, i["rec"])) for i in infos]
        if any(d is None for _, d in per) or set().union(*[set(d.channel) for _, d in per]) != set(CHANNELS):
            continue
        real_c3, real_mc = [], []
        n_c3 = n_mc = 0
        for info, d in per:
            on, off = cluster(d)
            keep = nap.in_n2(info, (on + off) / 2) if len(on) else np.zeros(0, bool)
            on, off = on[keep], off[keep]
            n_mc += len(on)
            real_mc.append((info["subject"], npt.score_nap(on, off, info["marks"], info["T"])))
            c3 = d[d.channel == "C3"]
            o3 = c3.onset.to_numpy(float)
            f3 = o3 + c3.duration.fillna(0).to_numpy(float)
            k3 = nap.in_n2(info, (o3 + f3) / 2)
            n_c3 += int(k3.sum())
            real_c3.append((info["subject"], npt.score_nap(o3[k3], f3[k3], info["marks"], info["T"])))
        out = {}
        for name, res in [("c3", real_c3), ("multi", real_mc)]:
            C, aff = npt.tensors(res, subjects)
            F, integ, P, R, AF = npt.summarise(C, aff)
            bF, binteg, *_ = npt.summarise(C, aff, W)
            tot = C.sum(0)
            out[name] = dict(F05=F[0, 1], integ=integ[0], P05=tot[1, 0] / max(tot[1, 0] + tot[1, 1], 1),
                             R05=tot[1, 0] / max(tot[1, 0] + tot[1, 2], 1), aff_F1=AF[0], boot=binteg)
        diff = out["multi"]["boot"] - out["c3"]["boot"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append(dict(detector=det, n_c3=n_c3, n_multi=n_mc,
                         **{f"{k}_c3": v for k, v in out["c3"].items() if k != "boot"},
                         **{f"{k}_multi": v for k, v in out["multi"].items() if k != "boot"},
                         d_integ=out["multi"]["integ"] - out["c3"]["integ"], d_lo=lo, d_hi=hi))
        print(det, "done", flush=True)
    r = pd.DataFrame(rows).sort_values("integ_multi", ascending=False)
    r.round(4).to_csv(paths.RESULTS / "nap_scoring" / "rater_rule.csv", index=False)
    pd.set_option("display.width", 220)
    print(r[["detector", "n_c3", "n_multi", "F05_c3", "F05_multi", "P05_c3", "P05_multi", "R05_c3", "R05_multi",
             "integ_c3", "integ_multi", "d_integ", "d_lo", "d_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
