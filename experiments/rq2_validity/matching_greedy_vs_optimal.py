"""
How much does the matching algorithm itself change F1? Greedy matching in detection
order (each detection takes the closest still-free label, as in most published
evaluation code) versus the maximum one-to-one matching used by experiments/rq1_benchmark/evaluate.py. Rules:
onset and IoU (DREAMS), and the centre rule that is the NAP main protocol; recordings
excluded in Mei et al. 2017 are left out.

    uv run python experiments/rq2_validity/matching_greedy_vs_optimal.py
"""
import json

import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.data.cache import POINT_MARKS
from spindlebench.metrics import events as em
from spindlebench.protocols.benchmark import anchor
from spindlebench import provenance  # noqa: E402
from spindlebench.protocols import benchmark  # noqa: E402

provenance.track(__file__, protocols=[benchmark.VERSION])


def greedy(det, ref, rule, tol):
    if not len(det) or not len(ref):
        return 0
    used = np.zeros(len(ref), bool)
    tp = 0
    for d in det[np.argsort(det[:, 0])]:
        if rule == "iou":
            s = em.iou_matrix(d[None], ref)[0]
            ok = (s >= tol - em.EPS) & ~used
            if ok.any():
                used[np.argmax(np.where(ok, s, -1))] = True
                tp += 1
        else:
            s = np.abs(ref[:, 0] - d[0])
            ok = (s <= tol + em.EPS) & ~used
            if ok.any():
                used[np.argmin(np.where(ok, s, np.inf))] = True
                tp += 1
    return tp


rows = []
m = pd.read_csv(paths.CACHE / "manifest.csv")
for _, r in m[~m.excluded & m.events.notna()].iterrows():  # label-free recordings (BOAS) have no events
    win = json.loads(r.windows)
    for event in r.events.split(","):
        for ann, (t0, t1) in win.items():
            lf = (paths.CACHE / "labels" / f"{r.dataset}/{r.rec}__{ann}__{event}.csv")
            if not lf.exists():
                continue
            ref = em.clip(pd.read_csv(lf)[["onset", "duration"]].to_numpy(float), t0, t1)
            for det_dir in sorted(((paths.RESULTS / "detections") / r.dataset).glob("*")):
                f = det_dir / f"{r.rec}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f)
                d = d[(d.channel.astype(str) == str(r.default_channel)) & (d.event == event)]
                det = em.clip(d[["onset", "duration"]].to_numpy(float), t0, t1)
                for rule, tol in [("onset", 0.5), ("onset", 1.0), ("center", 0.5), ("center", 1.0), ("iou", 0.2)]:
                    if rule == "iou" and r.dataset in POINT_MARKS:
                        continue
                    m_rule = "iou" if rule == "iou" else "onset"
                    da, ra = anchor(det, rule), anchor(ref, rule)
                    opt = em.match(da, ra, m_rule, tol)[0]
                    gr = greedy(da, ra, m_rule, tol)
                    rows.append(dict(dataset=r.dataset, rec=r.rec, detector=det_dir.name, reference=ann, event=event,
                                     rule=rule, tol=tol, n_det=len(det), n_ref=len(ref), tp_opt=opt, tp_greedy=gr))

df = pd.DataFrame(rows)
g = df.groupby(["dataset", "event", "rule", "tol"])[["n_det", "n_ref", "tp_opt", "tp_greedy"]].sum()
g["f1_opt"] = 2 * g.tp_opt / (g.n_det + g.n_ref)
g["f1_greedy"] = 2 * g.tp_greedy / (g.n_det + g.n_ref)
g["lost_tp_pct"] = 100 * (g.tp_opt - g.tp_greedy) / g.tp_opt
out = paths.RESULTS / "validity"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "greedy_vs_optimal_per_recording.csv", index=False)
g.to_csv(out / "greedy_vs_optimal.csv")
print(g.round(4).to_string())
worst = df.assign(diff=df.tp_opt - df.tp_greedy).sort_values("diff", ascending=False).head(5)
print("\nlargest per-recording differences:\n", worst.to_string(index=False))
