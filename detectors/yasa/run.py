"""
YASA spindle detector (Vallat & Walker 2021), reference adapter for the interface.

    .venv/Scripts/python detectors/yasa/run.py [--datasets ...] [--channels C3,F3,...]

Writes three configurations:
  yasa_default       published thresholds rel_pow 0.20, corr 0.65, rms 1.5
  yasa_loose         Update 1 thresholds 0.15 / 0.55 / 1.0
  yasa_dreams_tuned  best pooled F1 on the DREAMS sweep (results/dreams/sweep_expert1_onset.csv).
                     On NAP: the thresholds chosen on all 8 DREAMS excerpts (a transfer setting).
                     On DREAMS: leave-one-excerpt-out, i.e. each excerpt is detected with the
                     thresholds that are best on the other 7, so the DREAMS score is out of sample.
Broadband filter capped below Nyquist (DREAMS excerpt 3 is 50 Hz).
"""
import sys
from pathlib import Path

import pandas as pd

from spindlebench import paths, yasa_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

KEYS = ["rel_pow", "corr", "rms"]


def dreams_best(held_out=None):
    """thresholds with the best pooled F1 on the DREAMS sweep, leaving out excerpt `held_out`"""
    s = pd.read_csv(paths.RESULTS / "dreams/sweep_expert1_onset.csv")
    s = s[s.excerpt != held_out].groupby(KEYS)[["tp", "fp", "fn"]].sum()
    best = (2 * s.tp / (2 * s.tp + s.fp + s.fn)).idxmax()
    return dict(zip(KEYS, map(float, best)))


CONFIGS = {"yasa_default": yasa_utils.YASA_DEFAULT, "yasa_loose": yasa_utils.YASA_LOOSE, "yasa_dreams_tuned": dreams_best()}


def configs(row):
    if row.dataset.startswith("dreams"):
        return {**CONFIGS, "yasa_dreams_tuned": dreams_best(held_out=int(row.rec.removeprefix("excerpt")))}
    return CONFIGS

if __name__ == "__main__":
    args = common.parse_args(["dreams_spindles", "nap"])
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        out = {k: [] for k in CONFIGS}
        for ch in chans:
            x = data[ch_names.index(ch)].astype(float)
            for name, th in configs(row).items():
                for o, d in yasa_utils.run_yasa(x, sf, th):
                    out[name].append(dict(onset=o, duration=d, channel=ch, event="spindle", score=None))
        n = {name: common.write(args, row, name, rows) for name, rows in out.items()}
        print(row.dataset, row.rec, n, flush=True)
