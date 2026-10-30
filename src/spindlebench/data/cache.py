"""
Readers for the standardised cache (built by `spindlebench cache`, see data/export_cache.py).

    <cache>/manifest.csv                                one row per recording
    <cache>/labels/<dataset>/<rec>__hypno.csv           stages per epoch
    <cache>/labels/<dataset>/<rec>__<annotator>__<event>.csv   onset, duration
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.metrics.events import clip, intersection, union

POINT_MARKS = {"nap"}  # datasets whose labels are single time points (no durations, so no IoU)


def labels_dir():
    return paths.CACHE / "labels"


def manifest():
    return pd.read_csv(paths.CACHE / "manifest.csv")


def load_events(path):
    return pd.read_csv(path)[["onset", "duration"]].to_numpy(float) if path.exists() else None


def load_hypno(dataset, rec):
    """-> is_n2(times_s) for one recording"""
    h = pd.read_csv(labels_dir() / dataset / f"{rec}__hypno.csv")
    epoch = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    stages = h.stage.to_numpy()

    def is_n2(t):
        idx = np.clip((t // epoch).astype(int), 0, len(stages) - 1)
        return stages[idx] == "N2"
    return is_n2


def references(row, event):
    """annotator name -> (events, (t0, t1), has_durations); on DREAMS also the union and the
    intersection of the two experts (onsets within 1 s) on the window both covered"""
    win = json.loads(row.windows)
    refs = {}
    for ann in win:
        ev = load_events(labels_dir() / row.dataset / f"{row.rec}__{ann}__{event}.csv")
        if ev is not None:
            has_dur = row.dataset not in POINT_MARKS and not np.isnan(ev[:, 1]).any()
            refs[ann] = (ev, tuple(win[ann]), has_dur)
    if "expert1" in refs and "expert2" in refs:
        (e1, w1, _), (e2, w2, _) = refs["expert1"], refs["expert2"]
        w = (max(w1[0], w2[0]), min(w1[1], w2[1]))
        a, b = clip(e1, *w), clip(e2, *w)
        refs["union"] = (union(a, b), w, True)
        refs["intersection"] = (intersection(a, b), w, True)
    return refs


def signal_path(path, manifest_path=None):
    """file named in a manifest's `path` column. New manifests store it relative to the manifest's
    folder with forward slashes (portable); older ones relative to the repository root, with the
    separators of the machine that wrote them. Both resolve."""
    p = Path(str(path).replace("\\", "/"))
    if p.is_absolute():
        return p
    base = Path(manifest_path).parent if manifest_path else paths.CACHE
    return base / p if (base / p).exists() else paths.ROOT / p
