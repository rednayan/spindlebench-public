"""
The harmonised benchmark protocol (RQ1, RQ2): protocol `benchmark.v1`.

For every (dataset, recording, detector, event type) on the recording's default channel:
  reference   each annotator; on DREAMS also the union and intersection of the two experts
              (onsets within 1 s), scored on the window both experts covered
  scope       all  /  N2 (event midpoint inside an expert N2 epoch)
  protocol    rule x tolerance:
                onset   |det onset - ref onset|   <= 0.25 / 0.5 / 1.0 / 1.5 s
                center  |det centre - ref centre| <= 0.25 / 0.5 / 1.0 / 1.5 s
                iou     IoU >= 0.1 / 0.2 / 0.3 / 0.5   (references with durations only)
One-to-one maximum matching (metrics.events.match). The main protocol per dataset is MAIN.
"""
import numpy as np
import pandas as pd

from spindlebench.data.cache import POINT_MARKS, load_hypno, references  # noqa: F401  (POINT_MARKS re-exported)
from spindlebench.metrics.events import clip, in_mask, match, prf

VERSION = "benchmark.v1"
PROTOCOLS = [("onset", t) for t in (0.25, 0.5, 1.0, 1.5)] + \
            [("center", t) for t in (0.25, 0.5, 1.0, 1.5)] + \
            [("iou", t) for t in (0.1, 0.2, 0.3, 0.5)]
MAIN = {"dreams_spindles": ("expert1", "onset", 0.5), "dreams_kc": ("expert1", "onset", 0.5),
        "nap": ("expert", "center", 0.5)}


def anchor(ev, rule):
    """events re-anchored at their centre (NaN durations = a point mark, kept as is)"""
    if rule != "center" or not len(ev):
        return ev
    out = ev.copy()
    out[:, 0] += np.nan_to_num(out[:, 1]) / 2
    return out


_EVENT_TYPES = {}


def event_types(store, dataset, detector):
    """event types a detector produced anywhere in a dataset (an empty file says nothing)"""
    key = (str(store), dataset, detector)
    if key not in _EVENT_TYPES:
        files = (store / dataset / detector).glob("*.csv")
        _EVENT_TYPES[key] = set().union(*[set(pd.read_csv(f, usecols=["event"]).event) for f in files])
    return _EVENT_TYPES[key]


def evaluate(manifest, detectors, datasets, store):
    """per-recording counts for every detector x reference x scope x protocol"""
    rows = []
    for _, row in manifest[manifest.dataset.isin(datasets)].iterrows():
        is_n2 = load_hypno(row.dataset, row.rec)
        for det_name in detectors:
            f = store / row.dataset / det_name / f"{row.rec}.csv"
            if not f.exists():
                continue
            det_all = pd.read_csv(f)
            det_all = det_all[det_all.channel.astype(str) == str(row.default_channel)]
            for event in row.events.split(","):
                if event not in event_types(store, row.dataset, det_name):
                    continue  # this detector does not produce this event type
                det_ev = det_all[det_all.event == event][["onset", "duration"]].to_numpy(float)
                for ref_name, (ref, (t0, t1), has_dur) in references(row, event).items():
                    for scope in ("all", "N2"):
                        d, r = clip(det_ev, t0, t1), clip(ref, t0, t1)
                        if scope == "N2":
                            d, r = in_mask(d, is_n2), in_mask(r, is_n2)
                        for rule, tol in PROTOCOLS:
                            if rule == "iou" and not has_dur:
                                continue
                            m_rule = "iou" if rule == "iou" else "onset"
                            tp, fp, fn, _ = match(anchor(d, rule), anchor(r, rule), m_rule, tol)
                            p, rc, f1 = prf(tp, fp, fn)
                            rows.append(dict(dataset=row.dataset, rec=row.rec, excluded=row.excluded,
                                             detector=det_name, event=event, reference=ref_name, scope=scope,
                                             rule=rule, tol=tol, tp=tp, fp=fp, fn=fn,
                                             precision=p, recall=rc, f1=f1))
    return pd.DataFrame(rows)
