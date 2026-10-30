"""
Human ceiling: expert 2 scored against expert 1 on DREAMS (spindles, excerpts 1-6;
K-complexes, excerpts 1-5), under the same protocols as the detectors (experiments/rq1_benchmark/evaluate.py),
on the window both experts scored.

    uv run python experiments/raters/interrater.py      # -> results/validity/interrater.csv
"""

import pandas as pd

from spindlebench import paths
from spindlebench.metrics import events as em
from spindlebench.protocols import benchmark as ev
from spindlebench import provenance  # noqa: E402
from spindlebench.protocols import benchmark  # noqa: E402

provenance.track(__file__, protocols=[benchmark.VERSION])

m = pd.read_csv(paths.CACHE / "manifest.csv")
rows = []
for _, r in m[m.dataset.isin(["dreams_spindles", "dreams_kc"])].iterrows():
    event = r.events
    refs = ev.references(r, event)
    if "expert2" not in refs:
        continue
    (e1, w1, _), (e2, w2, _) = refs["expert1"], refs["expert2"]
    t0, t1 = max(w1[0], w2[0]), min(w1[1], w2[1])
    is_n2 = ev.load_hypno(r.dataset, r.rec)
    for scope in ("all", "N2"):
        a, b = em.clip(e2, t0, t1), em.clip(e1, t0, t1)
        if scope == "N2":
            a, b = em.in_mask(a, is_n2), em.in_mask(b, is_n2)
        for rule, tol in ev.PROTOCOLS:
            tp, fp, fn, _ = em.match(ev.anchor(a, rule), ev.anchor(b, rule), "iou" if rule == "iou" else "onset", tol)
            rows.append(dict(dataset=r.dataset, rec=r.rec, scope=scope, rule=rule, tol=tol, tp=tp, fp=fp, fn=fn))
df = pd.DataFrame(rows)
g = df.groupby(["dataset", "scope", "rule", "tol"])[["tp", "fp", "fn"]].sum()
g["F1_pooled"] = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
g["n_recordings"] = df.groupby(["dataset", "scope", "rule", "tol"]).rec.nunique()
out = paths.RESULTS / "validity"
out.mkdir(parents=True, exist_ok=True)
g.reset_index().to_csv(out / "interrater.csv", index=False)
print(g.reset_index().query("(rule == 'onset' and tol == 0.5) or (rule == 'iou' and tol == 0.2)")
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
