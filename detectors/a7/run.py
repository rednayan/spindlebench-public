"""
A7 spindle detector (Lacourse et al. 2019), Python port of the official MATLAB code.

    .venv/Scripts/python detectors/a7/run.py [--datasets dreams_spindles,nap] [--channels C3,F3,...]

Writes four configurations (published defaults otherwise, see README.md):
  a7                   every detection, as in the official EventDetection.txt; baseline = whole recording
  a7_incontext         only detections IN the spectral context (slow-ratio classifier, contextClassifier=1)
  a7_n2base            as a7, but the baseline is the expert-scored N2 (the published configuration);
  a7_incontext_n2base  uses expert staging, so never ranked (spindlebench.registry.USES_EXPERT_STAGING)
Preprocessing (as in the official pipeline / Snooz A7 tool): resample to 100 Hz
(polyphase; DREAMS excerpt 3 is upsampled 50 -> 100 Hz), zero-phase Butterworth
band-pass 0.3-30 Hz. The stage-free runs are the ranked ones, like every other benchmark detector.
"""
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import a7  # noqa: E402
import common  # noqa: E402

FS = a7.DEF["fs"]


def preprocess(x, sf):
    if sf != FS:
        fr = Fraction(FS / sf).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator)
    sos = butter(5, [0.3, 30], "bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x)


def n2_mask(dataset, rec, n):
    h = pd.read_csv(common.CACHE / "labels" / dataset / f"{rec}__hypno.csv")
    epoch = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    idx = np.clip((np.arange(n) / FS // epoch).astype(int), 0, len(h) - 1)
    m = h.stage.to_numpy()[idx] == "N2"
    return m if m.sum() > a7.DEF["bslLengthSec"] * FS else None  # no usable N2 -> whole recording


if __name__ == "__main__":
    args = common.parse_args(["dreams_spindles", "nap"])
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        t0 = time.time()
        out = {k: [] for k in ("a7", "a7_incontext", "a7_n2base", "a7_incontext_n2base")}
        for ch in chans:
            x = preprocess(data[ch_names.index(ch)].astype(float), sf)
            for tag, base in (("", None), ("_n2base", n2_mask(row.dataset, row.rec, len(x)))):
                events, _ = a7.a7_detect(x, base)
                for s, e, ctx in events:
                    r = dict(onset=s / FS, duration=(e - s + 1) / FS, channel=ch, event="spindle", score=None)
                    out["a7" + tag].append(r)
                    if ctx:
                        out["a7_incontext" + tag].append(r)
        n = {k: common.write(args, row, k, v) for k, v in out.items()}
        print(row.dataset, row.rec, n, f"{time.time() - t0:.1f}s", flush=True)
