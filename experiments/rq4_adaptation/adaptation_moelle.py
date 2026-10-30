"""
RQ4 rung for the best rule-based detector: recalibrate Mölle 2011's one threshold on NAP.

    uv run python experiments/rq4_adaptation/adaptation_moelle.py

Uses the paper-recipe implementation (detectors/moelle/run.py, stage-free variant) and sweeps its
threshold factor k (paper: 1.5 SD of the sigma-filtered signal). Same strategies, held-out recordings
and scoring as experiments/rq4_adaptation/adaptation_nap.py (C3, centre ±0.5 s vs expert, the 26 naps of the 13 subjects with
both naps included):
  default      k = 1.5 (the paper)
  population   k tuned on all other subjects (leave-one-subject-out)
  per_subject  k tuned on the same subject's other nap
  oracle       k tuned on the test nap itself (upper bound)
Writes results/adaptation/moelle_threshold_adaptation_{per_recording,summary}.csv.
"""
import json
import sys

import numpy as np
import pandas as pd

from spindlebench.data.cache import signal_path
from spindlebench import paths
from spindlebench.data.cache import load_hypno
from spindlebench.metrics import events as em
from spindlebench.protocols.benchmark import anchor

sys.path.insert(0, str(paths.ROOT / "detectors" / "moelle"))
from run import detect  # noqa: E402
from spindlebench import provenance  # noqa: E402
from spindlebench.protocols import benchmark  # noqa: E402

provenance.track(__file__, protocols=[benchmark.VERSION])

K = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
OUT = paths.RESULTS / "adaptation"


def sweep():
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    rows = []
    for row in m[(m.dataset == "nap") & ~m.excluded].itertuples():
        z = np.load(signal_path(row.path))
        ch = [str(c) for c in z["ch_names"]]
        x, sf = z["data"][ch.index(row.default_channel)].astype(float), float(z["sf"])
        t0, t1 = json.loads(row.windows)["expert"]
        ref = pd.read_csv(paths.CACHE / "labels" / f"nap/{row.rec}__expert__spindle.csv")[["onset", "duration"]].to_numpy(float)
        is_n2 = load_hypno("nap", row.rec)
        for k in K:
            on, du = detect(x, sf, k=k)
            det = np.c_[on, du]
            for scope in ("all", "N2"):
                d, r = em.clip(det, t0, t1), em.clip(ref, t0, t1)
                if scope == "N2":
                    d, r = em.in_mask(d, is_n2), em.in_mask(r, is_n2)
                tp, fp, fn, _ = em.match(anchor(d, "center"), anchor(r, "center"), "onset", 0.5)
                rows.append(dict(rec=row.rec, subject=int(row.rec[3:].split("_")[0]), day=int(row.rec[-1]),
                                 scope=scope, k=k, tp=tp, fp=fp, fn=fn))
        print(row.rec, flush=True)
    return pd.DataFrame(rows)


def best(df, fallback=1.5):
    g = df.groupby("k")[["tp", "fp", "fn"]].sum()
    f = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
    return fallback if f.isna().all() else float(f.idxmax())


def main():
    sw = sweep()
    rows = []
    for scope, s in sw.groupby("scope"):
        both = s.groupby("subject").day.nunique()
        for subj in both[both == 2].index:
            pop = best(s[s.subject != subj])
            for day in (1, 2):
                test = s[(s.subject == subj) & (s.day == day)]
                calib = s[(s.subject == subj) & (s.day != day)]
                for strat, k in {"default": 1.5, "population": pop, "per_subject": best(calib, pop),
                                 "oracle": best(test, pop)}.items():
                    r = test[np.isclose(test.k, k)]
                    rows.append(dict(scope=scope, subject=subj, day=day, strategy=strat, k=k,
                                     tp=int(r.tp.sum()), fp=int(r.fp.sum()), fn=int(r.fn.sum())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "moelle_threshold_adaptation_per_recording.csv", index=False)
    summ = res.groupby(["scope", "strategy"])[["tp", "fp", "fn"]].sum()
    summ["F1_pooled"] = 2 * summ.tp / (2 * summ.tp + summ.fp + summ.fn)
    summ["k_median"] = res.groupby(["scope", "strategy"]).k.median()
    summ = summ.reset_index()
    summ.to_csv(OUT / "moelle_threshold_adaptation_summary.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
