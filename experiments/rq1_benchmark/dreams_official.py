"""
The DREAMS assessment method (Devuyst et al. 2011) applied to every spindle detector on DREAMS.
It reproduces the published result (sensitivity 70.20%, false positives 26.44% of reference events)
for the detections shipped with the database.

    uv run python experiments/rq1_benchmark/dreams_official.py

The method is spindlebench.protocols.dreams_assessment (written from the published description; see
that module). Comparisons: every detector vs expert 1, expert 2, their union and intersection on
excerpts 1-6 (the two-expert excerpts, the authors' own protocol), and vs expert 1 on excerpts 1-8.
Whole recording, all stages, counts pooled over excerpts.
-> results/benchmark/dreams_official_protocol.csv
"""
import numpy as np
import pandas as pd

from spindlebench import paths, provenance
from spindlebench.data.cache import load_events
from spindlebench.protocols import dreams_assessment as da

provenance.track(__file__, protocols=[da.VERSION])

DATASET = "dreams_spindles"
STORE = paths.RESULTS / "detections" / DATASET
OUT = paths.RESULTS / "benchmark"
RUNS = [("1-6", range(1, 7), r) for r in ("union", "expert1", "expert2", "intersection")] + \
       [("1-8", range(1, 9), "expert1")]


def masks(row, detector):
    n = int(round(row.duration * row.sf))
    lab = paths.CACHE / "labels" / DATASET
    v1 = load_events(lab / f"{row.rec}__expert1__spindle.csv")
    v2 = load_events(lab / f"{row.rec}__expert2__spindle.csv")
    d = pd.read_csv(STORE / detector / f"{row.rec}.csv")
    d = d[(d.event == "spindle") & (d.channel.astype(str) == str(row.default_channel))]
    det = d[["onset", "duration"]].to_numpy(float)
    return (da.to_mask(det, row.sf, n), da.to_mask(v1, row.sf, n),
            None if v2 is None else da.to_mask(v2, row.sf, n))


def table(detectors=None):
    man = pd.read_csv(paths.CACHE / "manifest.csv")
    man = {r.rec: r for r in man[man.dataset == DATASET].itertuples()}
    detectors = detectors or sorted(p.name for p in STORE.iterdir() if p.is_dir())
    rows = []
    for det in detectors:
        cache = {n: masks(man[f"excerpt{n}"], det) for n in range(1, 9)}
        for label, excerpts, ref in RUNS:
            c = np.sum([da.count(*cache[n], ref) for n in excerpts], axis=0)
            m = da.measures(*c)
            rows.append(dict(detector=det, excerpts=label, reference=ref, TP=int(c[0]), FP=int(c[1]), FN=int(c[2]),
                             sens=m["sens"], fp_prop=m["fp_prop"], ppv=m["ppv"], f1=m["f1"]))
    return pd.DataFrame(rows)


def main():
    t = table()
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "dreams_official_protocol.csv", index=False)
    a = t[(t.detector == "dreams_auto") & (t.excerpts == "1-6") & (t.reference == "union")].iloc[0]
    print(f"DREAMS automatic detections vs union, excerpts 1-6: TP {a.TP}, FP {a.FP}, FN {a.FN}; "
          f"sensitivity {100 * a.sens:.2f}%, false positives {100 * a.fp_prop:.2f}% (published 70.20%, 26.44%)")


if __name__ == "__main__":
    main()
