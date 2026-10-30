"""
Classic spindle detectors as implemented in Wonambi (wonambi.detect.DetectSpindle),
run with each method's published default parameters.

    detectors/wonambi/.venv/Scripts/python detectors/wonambi/run.py [--datasets ...] [--channels ...] [--methods ...]

Own environment (Python 3.12, wonambi 7.17, scipy 1.12, numpy 1.26): wonambi still imports
scipy.signal.tukey, removed in scipy 1.13, so it cannot run in the main .venv.
Detector ids: wonambi_<method lower-case>, e.g. wonambi_ferrarelli2007.
Input: the cached µV signal at its native rate, demeaned by wonambi; no staging (whole
recording, like yasa_default). DREAMS excerpt 3 (50 Hz) is upsampled to 100 Hz first so
that no method's filter exceeds Nyquist (Ray2015 uses 0.3-35 Hz; see README.md).
"""
import argparse
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from wonambi.datatype import ChanTime  # noqa: E402
from wonambi.detect import DetectSpindle  # noqa: E402

METHODS = ["Ferrarelli2007", "Moelle2011", "Nir2011", "Wamsley2012", "Martin2013", "Ray2015",
           "Lacourse2018", "FASST", "FASST2", "UCSD", "Concordia"]
MIN_SF = 100.0


def chantime(x, sf, ch):
    d = ChanTime()
    d.s_freq = sf
    d.start_time = 0
    d.data = np.empty(1, dtype="O")
    d.data[0] = x[None, :]
    d.axis = {"chan": np.empty(1, dtype="O"), "time": np.empty(1, dtype="O")}
    d.axis["chan"][0] = np.array([ch], dtype="U")
    d.axis["time"][0] = np.arange(len(x)) / sf
    return d


if __name__ == "__main__":
    # strip --methods before handing the rest to the shared parser
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--methods", default=",".join(METHODS))
    extra, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = common.parse_args(["dreams_spindles", "nap"])
    methods = extra.methods.split(",")
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        out = {m: [] for m in methods}
        tt = {}
        for ch in chans:
            x = data[ch_names.index(ch)].astype(float)
            fs = sf
            if fs < MIN_SF:
                fr = Fraction(MIN_SF / fs).limit_denominator(100)
                x, fs = resample_poly(x, fr.numerator, fr.denominator), MIN_SF
            dat = chantime(x, fs, ch)
            for m in methods:
                t0 = time.time()
                try:
                    sp = DetectSpindle(method=m)(dat)
                    evs = sp.events
                except Exception as e:  # noqa: BLE001
                    print(f"  {m} failed on {row.rec} {ch}: {e!r}", flush=True)
                    evs = []
                tt[m] = tt.get(m, 0) + time.time() - t0
                for ev in evs:
                    out[m].append(dict(onset=ev["start"], duration=ev["end"] - ev["start"],
                                       channel=ch, event="spindle", score=None))
        n = {m: common.write(args, row, f"wonambi_{m.lower()}", rows) for m, rows in out.items()}
        print(row.dataset, row.rec, n, {m: round(t, 1) for m, t in tt.items()}, flush=True)
