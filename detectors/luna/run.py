"""
Luna SPINDLES (Purcell et al. 2017), through the lunapi Python bindings, with the program defaults.

    detectors/luna/.venv/Scripts/python detectors/luna/run.py [--datasets dreams_spindles,nap] [--channels C3,...]

Detector id: luna_spindles.
Defaults, as in luna-base spindles/spindles.cpp: one Morlet wavelet at fc 13.5 Hz with 7 cycles;
core threshold 4.5 x the whole-signal mean of the smoothed wavelet coefficient (th), flanking
threshold 2 (th2), duration 0.5-3 s, spindles closer than 0.5 s merged, Luna's own spindle QC.
Stage-free: run on the whole recording with no epoch mask, like the other stage-free rows.
The signal is the cached one in uV at its native rate, trimmed to whole seconds (Luna stores it
in 1 s EDF records).
"""
import sys
from pathlib import Path

import lunapi as lp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

DET = "luna_spindles"
COMMAND = "SPINDLES sig=EEG per-spindle"

if __name__ == "__main__":
    args = common.parse_args(["dreams_spindles", "nap"])
    proj = lp.proj()
    proj.silence(True)
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        rows = []
        sr = int(round(sf))
        for ch in chans:
            x = data[ch_names.index(ch)].astype(float)
            n = len(x) // sr
            inst = proj.empty_inst(f"{row.rec}_{ch}", nr=n, rs=1)
            inst.insert_signal("EEG", x[: n * sr], sr)
            t = inst.proc(COMMAND).get("SPINDLES: CH_F_SPINDLE")
            if t is None or not len(t):
                continue
            if "PASS" in t.columns:
                t = t[t.PASS.astype(float) == 1]
            for s, e in zip(t.START.astype(float), t.STOP.astype(float)):
                rows.append(dict(onset=s, duration=e - s, channel=ch, event="spindle", score=None))
        print(row.dataset, row.rec, common.write(args, row, DET, rows), flush=True)
