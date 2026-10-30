"""Reproduce the SUMOv2 DREAMS numbers (0.53 vs expert 1, 0.76 vs expert 2) with the protocol of
src/sumov2/scripts/final/dreams-f1-vs-experts.ipynb: N2 by event midpoint (expert 5 s hypnogram), IoU > 0.2,
greedy many-to-one matching, pooled over excerpts 1-8 (expert 1) / 1-6 (expert 2), full recording (no
1000 s window). Also compares our sumov2_n2 events one by one with the predictions shipped in the repo.
    .venv/Scripts/python.exe detectors/sumo/check_reproduction.py
"""
# Published SUMOv2 DREAMS protocol (src/sumov2/scripts/final/dreams-f1-vs-experts.ipynb), functions copied verbatim.
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; L = ROOT / "cache" / "labels" / "dreams_spindles"
SHIP = ROOT / "detectors/sumo/src/sumov2/output/annotations/dreams"
thr = [0.1, 0.2, 0.5]

def overlap(spindles, sleepstages):
    sleepstages = sleepstages == 2
    spindle_centers = (spindles[:, :2].sum(axis=1) // 2).astype(int)
    idx = [sleepstages[c // 5] if c // 5 < len(sleepstages) else False for c in spindle_centers]
    return spindles[idx]

def calc_tp_fp_fn(gt_spindles, pred_spindles, spindle_thresholds):
    tp, fp = [0] * len(spindle_thresholds), [0] * len(spindle_thresholds)
    used = [np.zeros(len(gt_spindles)) for _ in spindle_thresholds]
    gt_start = 0
    for pred_s in pred_spindles:
        inter = [0] * len(gt_spindles); uni = [0] * len(gt_spindles)
        for gi in range(gt_start, len(gt_spindles)):
            gs = gt_spindles[gi]
            if pred_s[1] < gs[0]: break
            if pred_s[0] > gs[1]:
                gt_start = gi + 1; continue
            inter[gi] = min(gs[1], pred_s[1]) - max(gs[0], pred_s[0])
            uni[gi] = max(gs[1], pred_s[1]) - min(gs[0], pred_s[0])
        ious = [inter[i] / uni[i] if uni[i] > 0 else 0 for i in range(len(inter))]
        for i, t in enumerate(spindle_thresholds):
            if len(ious) > 0 and max(ious) > t:
                tp[i] += 1; used[i][np.argmax(ious)] = 1
            else:
                fp[i] += 1
    fn = [len(gt_spindles) - np.sum(used[i]) for i in range(len(spindle_thresholds))]
    return tp, fp, fn

def stages(rec):
    h = pd.read_csv(L / f"{rec}__hypno.csv")
    assert h.epoch_start.iloc[1] == 5
    return np.where(h.stage.to_numpy() == "N2", 2, 0)

def gt(rec, ex):
    f = L / f"{rec}__{ex}__spindle.csv"
    d = pd.read_csv(f)
    return np.c_[d.onset, d.onset + d.duration]

def preds(src, rec):
    if src == "shipped":
        z = np.load(SHIP / f"{rec}_aggregated.npz"); return z[list(z.keys())[0]][:, :2]
    d = pd.read_csv(ROOT / "results" / "detections" / "dreams_spindles" / src / f"{rec}.csv")
    return np.c_[d.onset, d.onset + d.duration]

for src in ["shipped", "sumov2_n2", "sumov2", "sumo"]:
    for ex, recs in [("expert1", range(1, 9)), ("expert2", range(1, 7))]:
        T = np.zeros((3, 3))
        for r in recs:
            rec = f"excerpt{r}"; st = stages(rec)
            p = overlap(preds(src, rec), st)
            p = p[np.argsort(p[:, 0])]
            g = overlap(gt(rec, ex), st)
            T += np.array(calc_tp_fp_fn(g, p, thr)).T
        tp, fp, fn = T[1]
        print(f"{src:10s} {ex}  IoU>0.2  P {tp/(tp+fp):.3f} R {tp/(tp+fn):.3f} F1 {2*tp/(2*tp+fp+fn):.3f}  (tp {tp:.0f} fp {fp:.0f} fn {fn:.0f})")

# event-level agreement of our sumov2_n2 with shipped predictions
for r in range(1, 9):
    rec = f"excerpt{r}"; a = preds("shipped", rec); b = preds("sumov2_n2", rec)
    st = stages(rec); a2 = overlap(a, st); b2 = overlap(b, st)
    tp, fp, fn = calc_tp_fp_fn(a2[np.argsort(a2[:,0])], b2[np.argsort(b2[:,0])], [0.8])
    print(rec, "shipped", len(a), "ours", len(b), "| N2-mid: shipped", len(a2), "ours", len(b2), "IoU>0.8 matched", tp[0], "max |onset diff| of matched",
          np.round(np.max([np.min(np.abs(a2[:,0]-x)) for x in b2[:,0]]) if len(a2) and len(b2) else np.nan, 3))
