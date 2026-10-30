"""
The DREAMS assessment method (Devuyst et al. 2011, EMBC, pp. 1713-1716): protocol `dreams_assessment.v1`.

Written from the published description of the method and our own notes on its behaviour, not from
the DREAMS MATLAB code (which is distributed under CC BY-NC-ND 3.0 with the database). Its counts are
checked against the published result and against an earlier port of the original
(tests/test_dreams_assessment.py).

The method, as described:
  1. Each scorer's events (the detector, expert 1 = V1, expert 2 = V2) become sample intervals at the
     recording's sampling rate. Within one scorer, two consecutive events that touch or overlap are
     separated at their midpoint, leaving one empty sample between them.
  2. The scorers' intervals are laid on one time axis together; every maximal run of samples marked
     by at least one scorer is a cluster. Clusters are formed over all scorers at once, so an expert-2
     event can join a detection and an expert-1 event into one cluster.
  3. Two scorers agree in a cluster when their intervals share at least one sample (events that only
     touch do not). Each cluster has exactly one outcome, however many events it holds:
       the detector agrees with the reference           -> 1 TP
       otherwise, the detector is present               -> 1 FP
       otherwise, the reference is present              -> 1 FN
     The reference is V1, V2, V1 or V2 (union: the detector agrees with either expert), or V1 and V2
     (intersection: the experts agree with each other and the detector agrees with both).
  4. The whole recording and all sleep stages are scored. Counts are pooled over recordings.
  5. True negatives are approximated as round(duration / duration_event) - TP - FP - FN (duration_event
     = mean event length in seconds), so specificity and the false positive rate are estimates.
Reported measures: sensitivity TP/(TP+FN); false positive proportion FP/(TP+FN) (relative to the
number of reference events); PPV and F1 are added here.
"""
import numpy as np

VERSION = "dreams_assessment.v1"
REFERENCES = ("expert1", "expert2", "union", "intersection")


def to_mask(events, sf, n):
    """events (k, 2) [onset_s, duration_s] -> boolean sample mask of length n, with touching or
    overlapping events of this scorer separated by one empty sample at their midpoint"""
    m = np.zeros(n, bool)
    if not len(events):
        return m
    ev = np.asarray(events, float)
    ev = ev[np.argsort(ev[:, 0], kind="stable")]
    start = np.round(ev[:, 0] * sf).astype(int)
    stop = np.round((ev[:, 0] + ev[:, 1]) * sf).astype(int) - 1  # last sample (the end itself is excluded,
    #                                                                so events that only touch share no sample)
    for i in range(1, len(start)):
        if start[i] <= stop[i - 1] + 1:  # touches or overlaps the previous event
            mid = (stop[i - 1] + start[i]) // 2
            stop[i - 1], start[i] = mid - 1, mid + 1
    for a, b in zip(start, stop):
        a, b = max(a, 0), min(b, n - 1)
        if b >= a:
            m[a:b + 1] = True
    return m


def clusters(*masks):
    """[(first, last)] sample ranges of maximal runs where any mask is set"""
    any_ = np.logical_or.reduce(masks)
    d = np.diff(np.r_[0, any_.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def count(det, v1, v2, reference):
    """TP, FP, FN of a detector mask against the reference built from the expert masks v1, v2
    (v2 may be None when only one expert scored)"""
    if v2 is None:
        v2 = np.zeros_like(v1)
    tp = fp = fn = 0
    for a, b in clusters(det, v1, v2):
        d, e1, e2 = det[a:b + 1], v1[a:b + 1], v2[a:b + 1]
        if reference == "expert1":
            ref_present, agree = e1.any(), (d & e1).any()
        elif reference == "expert2":
            ref_present, agree = e2.any(), (d & e2).any()
        elif reference == "union":
            ref_present, agree = (e1 | e2).any(), (d & (e1 | e2)).any()
        elif reference == "intersection":  # the detector agrees with each expert; the experts agree
            ref_present, agree = (e1 & e2).any(), (d & e1).any() and (d & e2).any()
        else:
            raise ValueError(reference)
        if agree:
            tp += 1
        elif d.any():
            fp += 1
        elif ref_present:
            fn += 1
    return tp, fp, fn


def measures(tp, fp, fn, duration=None, duration_event=None):
    n_ref = tp + fn
    out = dict(sens=tp / n_ref if n_ref else np.nan, fp_prop=fp / n_ref if n_ref else np.nan,
               ppv=tp / (tp + fp) if tp + fp else np.nan,
               f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan)
    if duration is not None and duration_event:
        tn = round(duration / duration_event) - tp - fp - fn
        out.update(tn=tn, spec=tn / (tn + fp), fpr=fp / (tn + fp))
    return out
