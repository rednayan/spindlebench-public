"""
Does the NAP mark format (a single point, no onset/duration) drive the injection result?

    uv run python experiments/injection/sumo_inject_ft/mark_check.py

1. Where do marks sit inside spindles? For every N2 mark, the overlapping detection (detection
   interval +-0.25 s contains the mark) of each benchmark detector gives the mark's offset from the
   detection centre and its relative position (0 = onset, 1 = offset). Pooled and per nap.
2. Are the arm contrasts robust to the matching rule? Re-score every arm with
     centre +-0.5 s (primary), centre +-1.0 s, centre +-1.5 s, and
     containment: the mark lies inside the detection interval widened by 0.25 s on each side
     (placement-agnostic: any position of the mark inside the spindle counts).
   One-to-one Hungarian matching, N2 scope (mark and detection midpoint in N2), 30 included naps.
   Same person-level paired bootstrap and 3-seed averaging as report.py.
-> results/injection/benchmark/mark_position.csv, mark_rules.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "injection" / "benchmark"
SLACK = 0.25
BENCH_DETS = ["sumov2", "a7", "yasa_dreams_tuned", "moelle2011_faithful", "openspindlenet"]
import report as R  # noqa: E402  (ARMS, CONTRASTS, f1)
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)


def naps():
    m = pd.read_csv(ROOT / "cache" / "manifest.csv")
    return m[(m.dataset == "nap") & ~m.excluded.astype(bool)].rec.tolist()


def n2_fn(rec):
    h = pd.read_csv(ROOT / "cache" / "labels" / "nap" / f"{rec}__hypno.csv")
    ep = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    st = h.stage.to_numpy()
    return lambda t: st[np.clip((np.asarray(t) // ep).astype(int), 0, len(st) - 1)] == "N2"


def load_det(path):
    d = pd.read_csv(path)
    d = d[(d.channel.astype(str) == "C3") & (d.event == "spindle")]
    on = d.onset.to_numpy(float)
    return on, on + d.duration.fillna(0).to_numpy(float)


def counts(on, off, marks, rule):
    c = (on + off) / 2
    if not len(c) or not len(marks):
        return 0, len(c), len(marks)
    if rule == "contain":
        ok = (marks[None, :] >= on[:, None] - SLACK) & (marks[None, :] <= off[:, None] + SLACK)
    else:
        ok = np.abs(c[:, None] - marks[None, :]) <= float(rule) + 1e-9
    if not ok.any():
        return 0, len(c), len(marks)
    d = np.abs(c[:, None] - marks[None, :])
    r, k = linear_sum_assignment(np.where(ok, d, d[ok].sum() + 1))
    tp = int(ok[r, k].sum())
    return tp, len(c) - tp, len(marks) - tp


def mark_position():
    rows = []
    for rec in naps():
        n2 = n2_fn(rec)
        marks = pd.read_csv(ROOT / "cache" / "labels" / "nap" / f"{rec}__expert__spindle.csv").onset.to_numpy(float)
        marks = marks[n2(marks)]
        for d in BENCH_DETS:
            on, off = load_det(ROOT / "results" / "detections" / "nap" / d / f"{rec}.csv")
            for mk in marks:
                i = np.flatnonzero((on - SLACK <= mk) & (mk <= off + SLACK))
                if not len(i):
                    continue
                i = i[np.argmin(np.abs((on[i] + off[i]) / 2 - mk))]
                du = off[i] - on[i]
                rows.append(dict(rec=rec, detector=d, mark=mk, offset_from_centre=mk - (on[i] + off[i]) / 2,
                                 rel_pos=(mk - on[i]) / du if du > 0 else np.nan, duration=du))
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "mark_position.csv", index=False)
    q = lambda x: pd.Series({"n": len(x), "median_offset": x.offset_from_centre.median(),  # noqa: E731
                             "iqr_offset": x.offset_from_centre.quantile(.75) - x.offset_from_centre.quantile(.25),
                             "frac_|off|>0.5": (x.offset_from_centre.abs() > 0.5).mean(),
                             "median_rel_pos": x.rel_pos.median(),
                             "frac_first_quarter": (x.rel_pos < 0.25).mean(), "frac_last_quarter": (x.rel_pos > 0.75).mean(),
                             "median_dur": x.duration.median()})
    print("Mark position inside overlapping detections (N2 marks):")
    print(p.groupby("detector").apply(q, include_groups=False).round(3).to_string())
    per = p.groupby("rec").offset_from_centre.median()
    print(f"per-nap median offset from centre: min {per.min():+.2f}, max {per.max():+.2f} s, "
          f"naps with |median| > 0.25 s: {(per.abs() > 0.25).sum()} / {len(per)}")
    return p


def rules():
    recs = naps()
    units = sorted({r.split("_")[0] for r in recs})
    ui = np.array([units.index(r.split("_")[0]) for r in recs])
    store = ROOT / "results" / "injection" / "detections" / "nap"
    marks = {}
    for rec in recs:
        n2 = n2_fn(rec)
        mk = pd.read_csv(ROOT / "cache" / "labels" / "nap" / f"{rec}__expert__spindle.csv").onset.to_numpy(float)
        marks[rec] = (mk[n2(mk)], n2)
    rows, con = [], []
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=2000).astype(float)
    for rule in ["0.5", "1.0", "1.5", "contain"]:
        C = {}
        for arm, ids in R.ARMS.items():
            mats = []
            for i in ids:
                m = []
                for rec in recs:
                    on, off = load_det(store / i / f"{rec}.csv")
                    mk, n2 = marks[rec]
                    keep = n2((on + off) / 2)
                    m.append(counts(on[keep], off[keep], mk, rule))
                m = np.array(m, float)
                mats.append(np.stack([m[ui == u].sum(0) for u in range(len(units))]))
            C[arm] = np.stack(mats)
        point = {a: np.nanmean(R.f1(c.sum(1))) for a, c in C.items()}
        boot = {a: np.nanmean(R.f1(np.einsum("bu,suk->bsk", W, c)), axis=1) for a, c in C.items()}
        for a, c in C.items():
            tot = c.sum(1)
            rows.append(dict(rule=rule, arm=a, F1=point[a], P=np.mean(tot[:, 0] / (tot[:, 0] + tot[:, 1])),
                             R=np.mean(tot[:, 0] / (tot[:, 0] + tot[:, 2]))))
        for a, b in R.CONTRASTS[:5]:
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            con.append(dict(rule=rule, contrast=f"{a} - {b}", diff=point[a] - point[b], lo=lo, hi=hi,
                            excludes_0=bool(lo > 0 or hi < 0)))
    t, c = pd.DataFrame(rows), pd.DataFrame(con)
    t.round(4).to_csv(OUT / "mark_rules.csv", index=False)
    c.round(4).to_csv(OUT / "mark_rules_contrasts.csv", index=False)
    print(t.pivot_table(index="arm", columns="rule", values="F1", sort=False).round(3).to_string())
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    mark_position()
    rules()
