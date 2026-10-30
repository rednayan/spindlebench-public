"""
Shared code for the SUMOv2 injection experiment (plan: docs/thesis/plan_injection.md, section 3).
Runs inside detectors/sumo/.venv-sumov2 (Python 3.10, torch 2.5 CPU).

Preprocessing, model loading and post-processing are the zero-shot adapter's (../run.py):
  preprocess_sumov2   10th-order Butterworth HP 0.3 / LP 30 Hz (sosfiltfilt), resample to 100 Hz,
                      RobustScaler over the whole recording, clip +-20
  post-processing     p(spindle) > threshold (0.5 = published argmax), runs -> events,
                      merge_and_filter_spindles (0.3-2.5 s)
ftlib.py (the SUMO v1 experiment) is not imported: it puts SUMO v1's `sumo` package on the path.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repository root: experiments/<study>/<this folder>
SUMO_DIR = ROOT / "detectors" / "sumo"
REPO = SUMO_DIR / "src" / "sumov2"
WORK = ROOT / "results" / "injection" / "work"
DET = ROOT / "results" / "injection" / "detections"
FS = 100
HALF_WIDTH = 0.4                       # NAP real-label targets, as RQ4
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)
TOL = 0.5
N_VARIANTS = 4                         # independent injection draws per recording

sys.path.insert(0, str(SUMO_DIR))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "src" / "spindlebench" / "synth"))
sys.path.insert(0, str(ROOT / "detectors"))
import run as zs  # noqa: E402


def manifest():
    return pd.read_csv(ROOT / "cache" / "manifest.csv")


def recordings(dataset):
    m = manifest()
    return m[(m.dataset == dataset) & (~m.excluded.astype(bool))].reset_index(drop=True)


def prep_path(dataset, rec):
    return WORK / "prep" / dataset / f"{rec}.npz"


def load_prep(dataset, rec):
    z = np.load(prep_path(dataset, rec), allow_pickle=False)
    return {k: z[k] for k in z.files}


def to_mask(n, onsets, durations):
    m = np.zeros(n, np.int8)
    for on, du in zip(onsets, durations):
        a, b = int(round(on * FS)), int(round((on + du) * FS))
        m[max(a, 0):min(max(b, a + 1), n)] = 1
    return m


def resample_mask(mask, sf, n):
    """bool mask at sf -> n samples at 100 Hz (nearest)"""
    idx = np.clip((np.arange(n) * sf / FS).astype(int), 0, len(mask) - 1)
    return mask[idx]


def runs(mask, min_len):
    b = np.diff(np.r_[0, mask.astype(int), 0])
    s, e = np.argwhere(b == 1)[:, 0], np.argwhere(b == -1)[:, 0]
    return [(a, c) for a, c in zip(s, e) if c - a >= min_len]


# ------------------------------------------------------------------ model

def make_model(state_dict=None):
    model = zs.load_model(REPO)          # released SUMOv2 weights
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def predict_prob(model, y):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(torch.from_numpy(np.ascontiguousarray(y)).float()[None]), dim=1)[0]
    return prob[1].numpy()


def events(p1, thr):
    """(n, 3) onset s, offset s, score; SUMOv2 post-processing with argmax replaced by p1 > thr"""
    v = (p1 > thr).astype(int)
    d = np.diff(np.r_[0, v, 0])
    idx = np.c_[np.argwhere(d == 1), np.argwhere(d == -1)]
    ev = np.array([(a / FS, b / FS, float(p1[a:b].mean())) for a, b in idx], float).reshape(-1, 3)
    return zs.merge_and_filter_spindles(ev) if len(ev) else ev


def masked_gdl(logits, target, valid, smooth=1e-5):
    """SUMO's GeneralizedDiceLoss (Sudre 2017, weights 1/volume^2) restricted to valid samples.
    With valid == 1 everywhere it equals sumo.loss.GeneralizedDiceLoss exactly."""
    import torch
    import torch.nn.functional as F
    p = F.softmax(logits, dim=1)
    oh = F.one_hot(target.long(), num_classes=p.shape[1]).permute(0, 2, 1).float()
    v = valid.float()[:, None, :]
    inter = (v * p * oh).sum(2)
    union = (v * (p + oh)).sum(2)
    w = 1 / (v * oh).sum(2) ** 2
    w[torch.isinf(w)] = 1.0
    dice = (2 * (w * inter).sum(1) + smooth) / ((w * union).sum(1) + smooth)
    return (1 - dice).mean()


# ------------------------------------------------------------------ scoring

def match_count(det, ref, tol=TOL):
    from scipy.optimize import linear_sum_assignment
    if not len(det) or not len(ref):
        return 0
    d = np.abs(det[:, None] - ref[None, :])
    ok = d <= tol + 1e-9
    if not ok.any():
        return 0
    r, c = linear_sum_assignment(np.where(ok, d, d[ok].sum() + 1))
    return int(ok[r, c].sum())


def counts(p1, ref_times, keep, anchor="centre", thresholds=THRESHOLDS):
    """(n_thr, 3) tp, fp, fn. keep(t) -> bool array: which detection anchors count.
    ref_times must already be restricted to the scope."""
    out = []
    for thr in thresholds:
        ev = events(p1, thr)
        if len(ev):
            t = ev[:, 0] if anchor == "onset" else (ev[:, 0] + ev[:, 1]) / 2
            t = t[keep(t)]
        else:
            t = np.zeros(0)
        tp = match_count(t, ref_times)
        out.append((tp, len(t) - tp, len(ref_times) - tp))
    return np.array(out, float)


def best_threshold(c):
    tp, fp, fn = c.T
    den = 2 * tp + fp + fn
    if not (den > 0).any():
        return None, np.nan
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), -1)
    cand = np.flatnonzero(np.isclose(f1, f1.max()))
    k = cand[np.argmin(np.abs(THRESHOLDS[cand] - 0.5))]
    return float(THRESHOLDS[k]), float(f1[k])


def f1_at(c, thr=0.5):
    tp, fp, fn = c[np.argmin(np.abs(THRESHOLDS - thr))]
    return 2 * tp / max(2 * tp + fp + fn, 1)


def mask_keep(mask):
    return lambda t: mask[np.clip((np.asarray(t) * FS).astype(int), 0, len(mask) - 1)].astype(bool)


def real_scope(p, dataset):
    """(ref times, keep fn, anchor) for scoring against real expert labels, N2 scope
    (NAP: centre vs mark; DREAMS: onset vs expert-1 onset within 0-1000 s)"""
    n2 = p["n2"].astype(bool)
    if dataset == "nap":
        keep = mask_keep(n2)
        return p["marks"][keep(p["marks"])], keep, "centre"
    sc = n2 & (np.arange(len(n2)) < 1000 * FS)
    keep = mask_keep(sc)
    on = p["ref_on"]
    return on[keep(on + p["ref_du"] / 2) & (on + p["ref_du"] <= 1000)], (lambda t: keep(t)), "onset"


def inj_scope(p, v=0):
    """(ref times, keep fn, anchor) for scoring against injected events of variant v: detection
    centres count only inside the valid region (quiet N2 + injected events)"""
    ev = p[f"inj_ev{v}"]
    return ev[:, 0] + ev[:, 1] / 2, mask_keep(p[f"valid{v}"]), "centre"


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def write_detections(dataset, det_id, rec, ev, channel):
    d = DET / dataset / det_id
    d.mkdir(parents=True, exist_ok=True)
    rows = [dict(onset=round(a, 3), duration=round(b - a, 3), channel=channel, event="spindle", score=round(s, 4))
            for a, b, s in ev]
    pd.DataFrame(rows, columns=["onset", "duration", "channel", "event", "score"]).to_csv(d / f"{rec}.csv", index=False)
