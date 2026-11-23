"""
SUMOv2 preprocessing of every HMC and CinC 2018 night (plan 2.4):

    uv run python prepare.py [--workers 11] [--recs a,b]      (from experiments/student)

-> results/student/work/prep/<dataset>/<rec>.npz: x (float32, 100 Hz, whole night, default channel
   C3-M2), sf_native. detectors/sumo/run.py:preprocess_sumov2 unchanged: 10th-order Butterworth
   high-pass 0.3 Hz and low-pass 30 Hz (forward-backward), resample_poly to 100 Hz, RobustScaler over
   the whole night, clip ±20. The signal is not altered at artefacts (rejection is a loss mask).
Existing files are skipped.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import data
from spindlebench import provenance


def one(row):
    data.guard(row.dataset)
    out = data.prep_path(row.dataset, row.rec)
    if out.exists():
        return row.rec, None
    x, sf = data.load_signal(row)
    y = data.preprocess(x, sf)
    assert len(y) == int(np.ceil(len(x) * data.FS / sf)), (row.rec, len(y), len(x), sf)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, x=y, sf_native=sf)
    tmp.replace(out)
    return row.rec, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--recs", default=None)
    a = ap.parse_args()
    provenance.track(__file__, protocols=[])
    m = data.manifest()
    if a.recs:
        m = m[m.rec.isin(a.recs.split(","))]
    rows = [r for _, r in m.iterrows()]
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for rec, n in ex.map(one, rows):
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"{done}/{len(rows)} {rec} {n}", flush=True)


if __name__ == "__main__":
    main()
