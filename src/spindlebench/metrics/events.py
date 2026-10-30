"""
Event matching and scoring shared by every dataset and protocol.

Events are (n, 2) arrays of [onset_s, duration_s]. Matching is one-to-one and
maximises the number of matched pairs (Hungarian assignment), under one of:
  - "onset": |onset_det - onset_ref| <= tol   (default 0.5 s)
  - "iou":   IoU(det, ref) >= tol             (default 0.2, SUMO / SEED convention)
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RULES = {"onset": 0.5, "iou": 0.2}
EPS = 1e-9  # tolerance comparisons: a pair exactly tol apart counts despite float rounding


def empty():
    return np.zeros((0, 2))


def clip(ev, t0, t1):
    """events whose onset lies in [t0, t1)"""
    return ev[(ev[:, 0] >= t0) & (ev[:, 0] < t1)] if len(ev) else ev


def in_mask(ev, mask_fn):
    """events whose midpoint satisfies mask_fn(time_s) -> bool array (NaN duration = point mark)"""
    return ev[mask_fn(ev[:, 0] + np.nan_to_num(ev[:, 1]) / 2)] if len(ev) else ev


def iou_matrix(det, ref):
    s = np.maximum(det[:, None, 0], ref[None, :, 0])
    e = np.minimum(det[:, None, 0] + det[:, None, 1], ref[None, :, 0] + ref[None, :, 1])
    inter = np.clip(e - s, 0, None)
    union = det[:, None, 1] + ref[None, :, 1] - inter
    return np.where(inter > 0, inter / np.maximum(union, 1e-9), 0.0)


def match(det, ref, rule="onset", tol=None):
    """one-to-one matching; returns (tp, fp, fn, matched index pairs)"""
    tol = RULES[rule] if tol is None else tol
    if not len(det) or not len(ref):
        return 0, len(det), len(ref), []
    if rule == "onset":
        d = np.abs(det[:, None, 0] - ref[None, :, 0])
        ok = d <= tol + EPS
        cost = d
    else:
        m = iou_matrix(det, ref)
        ok = m >= tol - EPS
        cost = 1 - m
    # inadmissible pairs cost more than any full set of admissible ones,
    # so the assignment first maximises the number of admissible matches
    big = cost[ok].sum() + 1 if ok.any() else 1.0
    r, c = linear_sum_assignment(np.where(ok, cost, big))
    pairs = [(i, j) for i, j in zip(r, c) if ok[i, j]]
    tp = len(pairs)
    return tp, len(det) - tp, len(ref) - tp, pairs


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return p, r, f


def union(a, b, tol=1.0):
    """events in a, plus events in b whose onset is not within tol of any in a"""
    if not len(a):
        return b.copy()
    if not len(b):
        return a.copy()
    extra = b[np.abs(b[:, None, 0] - a[None, :, 0]).min(1) >= tol]
    out = np.vstack([a, extra])
    return out[np.argsort(out[:, 0])]


def intersection(a, b, tol=1.0):
    """events in a that b also marked (onset within tol)"""
    if not len(a) or not len(b):
        return empty()
    return a[np.abs(a[:, None, 0] - b[None, :, 0]).min(1) < tol]


def summarise(df, by, tp="tp", fp="fp", fn="fn"):
    """pooled (micro) and mean-of-recordings (macro) P/R/F1 per group. F1_mean is over the
    recordings where F1 is defined (n_f1); n_rec counts all recordings in the group"""
    rows = []
    for key, g in df.groupby(by, sort=False):
        P, R, F = prf(g[tp].sum(), g[fp].sum(), g[fn].sum())
        rows.append({**dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else (key,))),
                     "n_rec": len(g), "n_ref": int(g[tp].sum() + g[fn].sum()),
                     "n_det": int(g[tp].sum() + g[fp].sum()),
                     "P_pooled": P, "R_pooled": R, "F1_pooled": F,
                     "n_f1": int(g["f1"].notna().sum()), "F1_mean": g["f1"].mean(), "F1_sd": g["f1"].std()})
    return pd.DataFrame(rows)
