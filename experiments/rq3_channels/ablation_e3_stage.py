"""
E3 (RQ3), sleep-stage condition: what happens when a spindle detector is no longer
restricted to N2?

    uv run python experiments/rq3_channels/ablation_e3_stage.py

For every detector with full coverage of a dataset (default channel, main protocol:
DREAMS onset ±0.5 s vs expert 1 within its scored window; NAP centre ±0.5 s vs expert):
  - detections (TP / FP) are assigned to the expert sleep stage at the detection's
    midpoint, labels (matched / missed) to the stage at the label's midpoint
  - per stage: minutes scored, expert events per minute, detections per minute,
    precision (detection stage), recall (label stage)
This shows where precision is lost when the stage restriction is relaxed. The density
columns are descriptive only: density against the expert's own density separates
broken detectors from usable ones, but it does not rank the competitive ones, so it is
a coarse plausibility screen for unlabelled recordings, not an accuracy proxy.
Only spindle detectors are included (K-complex detectors emit no spindle events).
"""
import json

import numpy as np
import pandas as pd

from spindlebench import paths, registry
from spindlebench.metrics import events as em
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

MAIN = {"dreams_spindles": ("expert1", "onset"), "nap": ("expert", "center")}
STAGES = ["W", "N1", "N2", "N3", "REM"]
OUT = paths.RESULTS / "e3"


def centred(ev):
    out = ev.copy()
    if len(out):
        out[:, 0] += np.nan_to_num(out[:, 1]) / 2
    return out


m = pd.read_csv(paths.CACHE / "manifest.csv")
rows, minutes = [], []
for ds, (ann, rule) in MAIN.items():
    mm = m[(m.dataset == ds) & (~m.excluded)]
    dets = [p.name for p in sorted(((paths.RESULTS / "detections") / ds).glob("*"))
            if p.is_dir() and p.name not in registry.USES_EXPERT_STAGING and all((p / f"{r}.csv").exists() for r in mm.rec)
            and any((pd.read_csv(p / f"{r}.csv", usecols=["event"]).event == "spindle").any() for r in mm.rec)]
    for _, r in mm.iterrows():
        t0, t1 = json.loads(r.windows)[ann]
        h = pd.read_csv(paths.CACHE / "labels" / f"{ds}/{r.rec}__hypno.csv")
        epoch = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
        st = h.stage.to_numpy()
        stage_of = lambda t: st[np.clip((np.asarray(t) // epoch).astype(int), 0, len(st) - 1)]
        in_win = (h.epoch_start >= t0) & (h.epoch_start < t1)
        for s in STAGES:
            minutes.append(dict(dataset=ds, rec=r.rec, stage=s, minutes=((st == s) & in_win).sum() * epoch / 60))
        ref = em.clip(pd.read_csv(paths.CACHE / "labels" / f"{ds}/{r.rec}__{ann}__spindle.csv")[["onset", "duration"]].to_numpy(float), t0, t1)
        for det_name in dets:
            d = pd.read_csv((paths.RESULTS / "detections") / ds / det_name / f"{r.rec}.csv")
            d = d[(d.channel.astype(str) == str(r.default_channel)) & (d.event == "spindle")]
            det = em.clip(d[["onset", "duration"]].to_numpy(float), t0, t1)
            da, ra = (centred(det), centred(ref)) if rule == "center" else (det, ref)
            _, _, _, pairs = em.match(da, ra, "onset", 0.5)
            md = {i for i, _ in pairs}
            mr = {j for _, j in pairs}
            mid_d = centred(det)[:, 0] if len(det) else np.array([])
            mid_r = centred(ref)[:, 0] if len(ref) else np.array([])
            for i, s in enumerate(stage_of(mid_d)):
                rows.append(dict(dataset=ds, detector=det_name, rec=r.rec, stage=s, kind="TP" if i in md else "FP"))
            for j, s in enumerate(stage_of(mid_r)):
                rows.append(dict(dataset=ds, detector=det_name, rec=r.rec, stage=s, kind="TPref" if j in mr else "FN"))
            for j, s in enumerate(stage_of(mid_r)):
                rows.append(dict(dataset=ds, detector="EXPERT", rec=r.rec, stage=s, kind="REF")) if det_name == dets[0] else None

ev = pd.DataFrame(rows)
mins = pd.DataFrame(minutes).groupby(["dataset", "stage"]).minutes.sum()
cnt = ev.groupby(["dataset", "detector", "stage", "kind"]).size().unstack("kind", fill_value=0)
# every stage that was scored gets a row, also when nothing was detected/marked there
full = [(ds, det, st) for ds, det in cnt.reset_index()[["dataset", "detector"]].drop_duplicates().itertuples(index=False)
        for st in STAGES if mins.get((ds, st), 0) > 0]
cnt = cnt.reindex(cnt.index.union(pd.MultiIndex.from_tuples(full, names=cnt.index.names)), fill_value=0)
out = []
for (ds, det, s), c in cnt.iterrows():
    mn = mins.get((ds, s), 0)
    tp, fp, fn, ref = c.get("TP", 0), c.get("FP", 0), c.get("FN", 0), c.get("REF", 0)
    tpr = c.get("TPref", 0)
    out.append(dict(dataset=ds, detector=det, stage=s, minutes=mn,
                    per_min=((ref if det == "EXPERT" else tp + fp) / mn) if mn else np.nan,
                    precision=tp / (tp + fp) if tp + fp else np.nan,
                    recall=tpr / (tpr + fn) if tpr + fn else np.nan,
                    share_of_FP=np.nan, tp=tp, fp=fp, fn=fn))
res = pd.DataFrame(out)
fp_tot = res.groupby(["dataset", "detector"]).fp.transform("sum")
res["share_of_FP"] = res.fp / fp_tot
res["stage"] = pd.Categorical(res.stage, STAGES + ["?"])
res = res.sort_values(["dataset", "detector", "stage"])
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "stage_breakdown.csv", index=False)
pd.set_option("display.width", 220)
for ds in MAIN:
    r = res[res.dataset == ds]
    print(f"\n=== {ds}: events per minute by stage (EXPERT = reference density) ===")
    print(r.pivot(index="detector", columns="stage", values="per_min").round(2).to_string())
    print(f"--- {ds}: precision by stage ---")
    print(r[r.detector != "EXPERT"].pivot(index="detector", columns="stage", values="precision").round(2).to_string())
    print(f"--- {ds}: share of all false positives falling in each stage ---")
    print(r[r.detector != "EXPERT"].pivot(index="detector", columns="stage", values="share_of_FP").round(2).to_string())
print(f"\nwrote {OUT}/stage_breakdown.csv")
