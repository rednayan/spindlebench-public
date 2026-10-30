"""
Mölle et al. (2011) spindle detector, written from the paper (Sleep 34(10), Methods), as a check on
wonambi's implementation (`wonambi_moelle2011`), which differs in its filter and threshold base.

    python detectors/moelle/run.py [--datasets dreams_spindles,nap] [--channels C3,F3,...]

Recipe from the paper: band-pass at the individual fast-spindle peak (largest 12-15 Hz power) ±1.5 Hz
(FIR, zero phase); RMS in a sliding 200-ms window, smoothed
by a 200-ms moving average; a spindle is a run of the smoothed RMS above 1.5 SD of the band-passed
signal lasting 0.5-3 s. The paper takes the peak and the SD over NonREM sleep of the recording.

  moelle2011_faithful       SD over the whole recording: stage-free, ranked with the other detectors
  moelle2011_faithful_nrem  SD over expert-scored N2+N3, as in the paper; uses expert staging, so it
                            is a reproduction run and is never ranked (spindlebench.registry.USES_EXPERT_STAGING)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import filtfilt, firwin, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common

LABELS = common.CACHE / "labels"


def nrem_mask(dataset, rec, n, sf):
    h = pd.read_csv(LABELS / dataset / f"{rec}__hypno.csv")
    ep = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    idx = np.clip((np.arange(n) / sf // ep).astype(int), 0, len(h) - 1)
    return h.stage.isin(["N2", "N3"]).to_numpy()[idx]


def peak_freq(x, sf, base):
    """individual fast-spindle peak: the largest power in 12-15 Hz of the baseline's Welch spectrum"""
    seg = x if base is None or base.sum() < 60 * sf else x[base]  # < 1 min NREM: whole recording
    f, p = welch(seg, sf, nperseg=int(4 * sf))
    k = (f >= 12) & (f <= 15)
    return float(f[k][np.argmax(p[k])])


def detect(x, sf, base=None, k=1.5):
    """base: boolean mask of the samples the peak and SD are taken over (None = whole recording);
    k: threshold in SDs (the paper's 1.5; experiments/rq4_adaptation/adaptation_moelle.py recalibrates it)"""
    x = x - x.mean()
    pk = peak_freq(x, sf, base)
    f = filtfilt(firwin(int(sf) | 1, [pk - 1.5, pk + 1.5], pass_zero=False, fs=sf), [1.0], x)
    w = int(round(0.2 * sf))
    rms = uniform_filter1d(np.sqrt(uniform_filter1d(f ** 2, w, mode="nearest")), w, mode="nearest")
    thr = k * (f if base is None or base.sum() < 60 * sf else f[base]).std()
    edges = np.diff(np.r_[0, (rms > thr).astype(int), 0])
    on, off = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    dur = (off - on) / sf
    keep = (dur >= 0.5) & (dur <= 3.0)
    return on[keep] / sf, dur[keep]


if __name__ == "__main__":
    args = common.parse_args(["dreams_spindles", "nap"])
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        out = {"moelle2011_faithful": [], "moelle2011_faithful_nrem": []}
        base = nrem_mask(row.dataset, row.rec, data.shape[1], sf)
        for ch in chans:
            x = data[ch_names.index(ch)].astype(float)
            for name, b in (("moelle2011_faithful", None), ("moelle2011_faithful_nrem", base)):
                for o, d in zip(*detect(x, sf, b)):
                    out[name].append(dict(onset=o, duration=d, channel=ch, event="spindle", score=None))
        n = {name: common.write(args, row, name, rows) for name, rows in out.items()}
        print(row.dataset, row.rec, n, flush=True)
