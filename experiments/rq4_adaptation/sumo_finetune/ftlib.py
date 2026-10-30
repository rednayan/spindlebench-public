"""
Shared code for the RQ4 SUMO fine-tuning experiment (runs inside detectors/sumo/.venv-sumo).

Preprocessing, model and post-processing are reused from ../run.py (the zero-shot adapter):
  preprocess_sumo  10th-order Butterworth HP 0.3 Hz + LP 30 Hz (sosfiltfilt, padlen 30),
                   polyphase resampling to 100 Hz, z-score over the whole recording
  model            SUMO U-Net, forward() = dense logits + 0.42 s moving average
  post-processing  softmax; spindle where p(spindle) > threshold (0.5 == the published argmax);
                   each run of ones is one event; score = mean p(spindle) over the event

Scoring (for threshold choice and early stopping) re-implements the evaluator's main NAP protocol
(experiments/rq1_benchmark/evaluate.py + spindlebench.metrics.events.match): detection centre vs expert mark, |d| <= 0.5 s, one-to-one
Hungarian matching maximising the number of pairs, scope N2 = event midpoint in an expert N2 epoch.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root: experiments/<study>/<this folder>
SUMO_DIR = ROOT / "detectors" / "sumo"
REPO = SUMO_DIR / "src" / "sumo"
WORK = ROOT / "results" / "rq4" / "work"
DET = ROOT / "results" / "rq4" / "detections"
FS = 100
HALF_WIDTH = 0.4                      # label option A: mask = 1 in [mark - 0.4 s, mark + 0.4 s]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)
TOL = 0.5

sys.path.insert(0, str(SUMO_DIR))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "detectors"))
import common  # noqa: E402  (signal_path: manifest paths in either format)
import run as zs  # noqa: E402  (the zero-shot adapter: filters, preprocess_sumo, load_model)


# ------------------------------------------------------------------ data

def manifest():
    m = pd.read_csv(ROOT / "cache" / "manifest.csv")
    return m


def nap_included():
    m = manifest()
    m = m[(m.dataset == "nap") & (~m.excluded.astype(bool))].reset_index(drop=True)
    return m


def prep_path(dataset, rec):
    return WORK / "prep" / dataset / f"{rec}.npz"


def prepare_recording(row):
    """preprocessed signal @100 Hz on the default channel (+ for NAP: target mask, N2 mask, marks)"""
    out = prep_path(row.dataset, row.rec)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(common.signal_path(row.path, ROOT / "cache" / "manifest.csv"))
    ch = [str(c) for c in z["ch_names"]]
    x = z["data"][ch.index(row.default_channel)].astype(float)
    sf = float(z["sf"])
    y = zs.preprocess_sumo(x, sf).astype(np.float32)
    n = len(y)
    h = pd.read_csv(ROOT / "cache" / "labels" / row.dataset / f"{row.rec}__hypno.csv")
    epoch = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    t = np.arange(n) / FS
    idx = np.clip((t // epoch).astype(int), 0, len(h) - 1)
    n2 = (h.stage.to_numpy()[idx] == "N2")
    extra = {}
    if row.dataset == "nap":
        marks = pd.read_csv(ROOT / "cache" / "labels" / "nap" / f"{row.rec}__expert__spindle.csv").onset.to_numpy(float)
        mask = np.zeros(n, np.int8)
        for mk in marks:
            a, b = int(np.ceil((mk - HALF_WIDTH) * FS)), int(np.floor((mk + HALF_WIDTH) * FS)) + 1
            mask[max(a, 0):min(b, n)] = 1
        extra = dict(mask=mask, marks=marks)
    np.savez_compressed(out, y=y, n2=n2, epoch=epoch, stages=h.stage.to_numpy().astype(str), **extra)


def load_prep(dataset, rec):
    z = np.load(prep_path(dataset, rec), allow_pickle=False)
    return {k: z[k] for k in z.files}


def n2_runs(n2, min_len):
    """contiguous N2 sample runs (start, stop) of at least min_len samples"""
    b = np.diff(np.r_[0, n2.astype(int), 0])
    s, e = np.argwhere(b == 1)[:, 0], np.argwhere(b == -1)[:, 0]
    return [(a, c) for a, c in zip(s, e) if c - a >= min_len]


# ------------------------------------------------------------------ model

def get_config():
    from sumo.config import Config
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        return Config("predict", create_dirs=False)  # default.yaml: SUMO's training config
    finally:
        os.chdir(cwd)


def make_model(init="released", state_dict=None):
    import torch
    from sumo.model import SUMO
    model = SUMO(get_config())
    if state_dict is not None:
        model.load_state_dict(state_dict)
    elif init == "released":
        ckpt = torch.load(REPO / "output" / "final.ckpt", map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    return model


def predict_prob(model, y):
    """p(spindle) per sample @100 Hz, same computation as run.predict"""
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(torch.from_numpy(np.ascontiguousarray(y)).float()[None]), dim=1)[0]
    return prob[1].numpy()


def events(p1, thr):
    """(n, 3) onset s, offset s, score -- run.detect with argmax replaced by p1 > thr"""
    vect = (p1 > thr).astype(int)
    diff = np.diff(np.r_[0, vect, 0])
    idx = np.c_[np.argwhere(diff == 1), np.argwhere(diff == -1)]
    return np.array([(a / FS, b / FS, float(p1[a:b].mean())) for a, b in idx], float).reshape(-1, 3)


def write_detections(dataset, det_id, rec, ev, channel):
    d = DET / dataset / det_id
    d.mkdir(parents=True, exist_ok=True)
    rows = [dict(onset=round(on, 3), duration=round(off - on, 3), channel=channel, event="spindle",
                 score=round(sc, 4)) for on, off, sc in ev]
    pd.DataFrame(rows, columns=["onset", "duration", "channel", "event", "score"]).to_csv(d / f"{rec}.csv", index=False)


# ------------------------------------------------------------------ scoring (evaluator's NAP protocol)

def match_count(det_c, ref_c, tol=TOL):
    from scipy.optimize import linear_sum_assignment
    if not len(det_c) or not len(ref_c):
        return 0
    d = np.abs(det_c[:, None] - ref_c[None, :])
    ok = d <= tol + 1e-9  # a pair exactly tol apart counts despite float rounding (as spindlebench.metrics.events.EPS)
    if not ok.any():
        return 0
    big = d[ok].sum() + 1
    r, c = linear_sum_assignment(np.where(ok, d, big))
    return int(ok[r, c].sum())


def counts_over_thresholds(p1, prep, scope="N2", thresholds=THRESHOLDS):
    """array (n_thr, 3) tp, fp, fn on one nap"""
    marks = prep["marks"]
    stages, epoch = prep["stages"], float(prep["epoch"])

    def keep(t):
        if scope == "all":
            return np.ones(len(t), bool)
        i = np.clip((t // epoch).astype(int), 0, len(stages) - 1)
        return stages[i] == "N2"
    ref = marks[keep(marks)]
    out = []
    for thr in thresholds:
        ev = events(p1, thr)
        c = (ev[:, 0] + ev[:, 1]) / 2 if len(ev) else np.zeros(0)
        # the evaluator re-anchors at onset + round(duration)/2; rounding to 10 ms is exact at 100 Hz
        c = c[keep(c)]
        tp = match_count(c, ref)
        out.append((tp, len(c) - tp, len(ref) - tp))
    return np.array(out, float)


def best_threshold(counts):
    """counts: (n_thr, 3) pooled over calibration naps -> (thr, F1); ties -> closest to 0.5;
    None if no event or mark at all (e.g. a nap without N2)"""
    tp, fp, fn = counts.T
    den = 2 * tp + fp + fn
    if not (den > 0).any():
        return None, np.nan
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), -1)
    cand = np.flatnonzero(np.isclose(f1, f1.max()))
    k = cand[np.argmin(np.abs(THRESHOLDS[cand] - 0.5))]
    return float(THRESHOLDS[k]), float(f1[k])


def f1_of(counts):
    tp, fp, fn = np.asarray(counts, float).T
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)


def folds():
    return pd.read_csv(WORK / "folds.csv")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
