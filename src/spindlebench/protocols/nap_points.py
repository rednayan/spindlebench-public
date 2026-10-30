"""
Scoring detections against point marks (NAP): protocol `nap_points.v1`.

Documented methods only (decision 2026-10-28). Marks are single points (no onset, no duration).

1. Tolerance matching + tolerance sweep (primary).
   One-to-one (Hungarian) matching of detection centre to mark within +-tau, as in ECG beat matching
   (ANSI/AAMI EC57, bxb), SpikeForest and TCPDBench. tau on a grid fixed in advance:
   0.25, 0.50, ..., 1.50 s. Pooled F1 at each tau, and the integrated F1 = area under F1(tau) over
   [0.25, 1.5] s divided by its width (trapezoid), following the "F1 integrated over thresholds" of
   SUMO (Kaulen et al. 2022) and the area-under-curve summaries of Kim et al. (AAAI 2022).
2. Affiliation precision / recall (Huet et al., KDD 2022), computed with the authors' library
   (affiliation-metrics-py, commit 8d84498, MIT). The library rejects zero-length events; following
   the paper's Appendix B (a point at t is the limit of [t, t+eps)), each mark is passed as
   [t, t + 0.01 s]. Per recording: Trange = (0, end of the recording); affiliation P and R are
   averaged over recordings (P over recordings where it is defined); affiliation F1 is their
   harmonic mean.
3. Containment check (not primary): the mark lies inside the detection widened by 0.25 s on each
   side, one-to-one ("pointing game"-style hit, Zhang et al. ECCV 2016).

Random baseline (Kim et al. 2022; Huet et al. 2022): the same number of detections with the same
durations placed uniformly at random (midpoint in N2, non-overlapping).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

VERSION = "nap_points.v1"
TAUS = np.round(np.arange(0.25, 1.501, 0.25), 2)
MARGIN = 0.25
EPS_POINT = 0.01
KEYS = [f"tau{t}" for t in TAUS] + ["contain"]


def match_count(ok, cost):
    if not ok.any():
        return 0
    r, c = linear_sum_assignment(np.where(ok, cost, cost[ok].sum() + 1))
    return int(ok[r, c].sum())


def score_nap(on, off, marks, T):
    """-> dict: per-tau (tp, fp, fn), containment (tp, fp, fn), affiliation (P, R)"""
    from affiliation.metrics import pr_from_events
    nd, nm = len(on), len(marks)
    c = (on + off) / 2
    out = {}
    if nd and nm:
        d = np.abs(c[:, None] - marks[None, :])
        for tau in TAUS:
            tp = match_count(d <= tau + 1e-9, d)
            out[f"tau{tau}"] = (tp, nd - tp, nm - tp)
        ok = (marks[None, :] >= on[:, None] - MARGIN) & (marks[None, :] <= off[:, None] + MARGIN)
        tp = match_count(ok, d)
        out["contain"] = (tp, nd - tp, nm - tp)
    else:
        for tau in TAUS:
            out[f"tau{tau}"] = (0, nd, nm)
        out["contain"] = (0, nd, nm)
    # affiliation: disjoint ordered predictions (merge overlaps), marks as [t, t+eps]
    if nm:
        pred = []
        for a, b in zip(on, np.maximum(off, on + EPS_POINT)):
            if pred and a <= pred[-1][1]:
                pred[-1] = (pred[-1][0], max(pred[-1][1], float(b)))
            else:
                pred.append((float(a), float(b)))
        gt = []
        for t in marks:
            if not gt or t > gt[-1][1]:
                gt.append((float(t), float(t) + EPS_POINT))
        Tr = (0.0, max(T, pred[-1][1] if pred else 0, gt[-1][1]) + 1.0)
        if pred:
            r = pr_from_events(pred, gt, Tr)
            out["aff"] = (r["precision"], r["recall"])
        else:
            out["aff"] = (np.nan, 0.0)
    else:
        out["aff"] = (np.nan, np.nan)
    return out


def random_dets(on, off, info, rng):
    """same count and durations, uniform midpoints in N2, non-overlapping"""
    du = off - on
    ep = info["epoch"]
    epochs = info["n2_epochs"]
    if not len(du) or not len(epochs):
        return np.zeros(0), np.zeros(0)
    ro, rf = [], []
    for dd in rng.permutation(du):
        for _ in range(200):
            mid = (rng.choice(epochs) + rng.random()) * ep
            a, b = mid - dd / 2, mid + dd / 2
            if a >= 0 and all(b <= x or a >= y for x, y in zip(ro, rf)):
                ro.append(a), rf.append(b)
                break
    o = np.argsort(ro)
    return np.array(ro)[o], np.array(rf)[o]


def f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)


def integrated(f):
    """f: (..., len(TAUS)) -> trapezoid area / width"""
    return np.trapezoid(f, TAUS, axis=-1) / (TAUS[-1] - TAUS[0])


def tensors(rows, subjects):
    """rows of one detector -> counts (n_subj, n_keys, 3), aff P/R per nap with subject index"""
    C = np.zeros((len(subjects), len(KEYS), 3))
    aff = []
    for subj, sc in rows:
        i = subjects.index(subj)
        for k, key in enumerate(KEYS):
            C[i, k] += sc[key]
        aff.append((i, *sc["aff"]))
    return C, np.array(aff, float)


def summarise(C, aff, W=None):
    """point estimates or (with W: bootstrap weights (B, n_subj)) arrays of
    F1 per key, integrated F1, affiliation P, R, F"""
    if W is None:
        W = np.ones((1, C.shape[0]))
    tot = np.einsum("bs,skc->bkc", W, C)
    F = f1(tot[..., 0], tot[..., 1], tot[..., 2])
    integ = integrated(F[:, :len(TAUS)])
    w_nap = W[:, aff[:, 0].astype(int)]
    P = np.nansum(w_nap * np.nan_to_num(aff[:, 1]), 1) / np.sum(w_nap * ~np.isnan(aff[:, 1]), 1)
    R = np.nansum(w_nap * np.nan_to_num(aff[:, 2]), 1) / np.sum(w_nap * ~np.isnan(aff[:, 2]), 1)
    AF = 2 * P * R / (P + R)
    return F, integ, P, R, AF
