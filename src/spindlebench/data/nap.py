"""NAP recordings from the cache: included naps, point marks, N2 scope, C3 detections."""
import numpy as np
import pandas as pd

from spindlebench import paths


def naps():
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    return m[(m.dataset == "nap") & ~m.excluded.astype(bool)].reset_index(drop=True)


def nap_info(row):
    h = pd.read_csv(paths.CACHE / "labels" / "nap" / f"{row.rec}__hypno.csv")
    ep = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    st = h.stage.to_numpy()
    marks = pd.read_csv(paths.CACHE / "labels" / "nap" / f"{row.rec}__expert__spindle.csv").onset.to_numpy(float)
    in_n2 = lambda t: st[np.clip((np.asarray(t, float) // ep).astype(int), 0, len(st) - 1)] == "N2"  # noqa: E731
    n2_epochs = np.flatnonzero(st == "N2")
    return dict(rec=row.rec, subject=row.rec.split("_")[0], marks=np.sort(marks[in_n2(marks)]), stages=st, epoch=ep,
                n2_epochs=n2_epochs, T=max(float(row.duration), len(st) * ep))


def in_n2(info, t):
    st, ep = info["stages"], info["epoch"]
    return st[np.clip((np.asarray(t, float) // ep).astype(int), 0, len(st) - 1)] == "N2"


def load_det(path, info):
    d = pd.read_csv(path)
    d = d[(d.channel.astype(str) == "C3") & (d.event == "spindle")]
    on = d.onset.to_numpy(float)
    off = on + d.duration.fillna(0).to_numpy(float)
    keep = in_n2(info, (on + off) / 2)
    o = np.argsort(on[keep])
    return on[keep][o], off[keep][o]
