"""
Rule-based K-complex detectors.

    .venv/Scripts/python detectors/kcomplex_rules/run.py [--datasets dreams_kc,nap] [--channels C3,...]
                                                          [--detectors spinky_kc,yasa_sw_kc] [--workers 1]

  spinky_kc   Python port of Spinky's K-complex detector (Lajnef et al. 2017), TQWT-MCA transient
              component + trough threshold. Spinky has no fixed default threshold (it is calibrated per
              subject on scored pages; the paper's one example value is -60 µV). We use -35 µV, our
              reading of the per-subject optima plotted for MASS (Fig. 5B, no numbers given). Not tuned
              on our labels; sensitivity -25 to -60 µV: threshold_sweep.py.
  yasa_sw_kc  NOT a published K-complex detector: YASA's slow-wave detector (sw_detect) with its
              published defaults (0.3-1.5 Hz, neg. 40-200 µV, p-p 75-350 µV, neg. 0.3-1.5 s,
              pos. 0.1-1 s); its 75 µV peak-to-peak rule is the AASM slow-wave criterion (the AASM K-complex
              definition has no amplitude threshold), used here as a proxy.
              Each detected negative-positive wave is written as a kcomplex.
Both run on the whole recording without staging (as yasa_default does for spindles).
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import spinky as sk  # noqa: E402

SPINKY_THRESHOLD = -35.0
DETECTORS = ["spinky_kc", "yasa_sw_kc"]


def run_spinky(x, sf, ch):
    d = sk.DEF
    L = int(round(d["epoch_s"] * sf))
    n_ep = len(x) // L                       # data_epoching.m drops the last incomplete epoch
    epochs = x[:n_ep * L].reshape(n_ep, L)
    transit, _ = sk.signal_decomposition(epochs, sf)
    delay = d["kc_order"] / 2                # linear-phase FIR group delay (samples), see README
    rows = []
    for e in range(n_ep):
        troughs, k = sk.kc_detection(transit[e], SPINKY_THRESHOLD, sf)
        for i in troughs:
            s, en = sk.kc_extent(k, i)
            onset = (e * L + s - delay) / sf
            rows.append(dict(onset=max(onset, 0.0), duration=(en - s + 1) / sf, channel=ch,
                             event="kcomplex", score=-float(k[i])))
    return rows


def run_yasa_sw(x, sf, ch):
    import yasa
    sw = yasa.sw_detect(x, sf, hypno=None, verbose="ERROR")
    if sw is None:
        return []
    s = sw.summary()
    return [dict(onset=a, duration=b, channel=ch, event="kcomplex", score=None)
            for a, b in s[["Start", "Duration"]].to_numpy()]


FUNCS = {"spinky_kc": run_spinky, "yasa_sw_kc": run_yasa_sw}


def job(task):
    det, x, sf, ch = task
    t0 = time.time()
    return det, ch, FUNCS[det](x, sf, ch), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--workers", type=int, default=1)
    extra, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = common.parse_args(["dreams_kc", "nap"])
    dets = extra.detectors.split(",")
    pool = ProcessPoolExecutor(extra.workers) if extra.workers > 1 else None
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        tasks = [(d, data[ch_names.index(ch)].astype(float), sf, ch) for d in dets for ch in chans]
        results = pool.map(job, tasks) if pool else map(job, tasks)
        out, tt = {d: [] for d in dets}, {d: 0.0 for d in dets}
        for det, ch, rows, dt in results:
            out[det].extend(rows)
            tt[det] += dt
        n = {d: common.write(args, row, d, rows) for d, rows in out.items()}
        print(row.dataset, row.rec, n, {d: round(t, 1) for d, t in tt.items()}, flush=True)
