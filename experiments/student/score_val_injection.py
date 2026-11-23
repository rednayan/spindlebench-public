"""
Scoring on the injected validation nights (plan 6.2, 6.3). No expert label anywhere.

    uv run python score_val_injection.py teachers [--variants 0,1,2,3]     6.3 identifiability pre-check

Matching (as experiments/injection/score_bench.py): detection centre vs injected-event centre within
+-0.5 s, one-to-one maximum matching. Detections are considered only where their centre lies in the
scored region = quiet N2 (where no teacher fired on the original night) or inside an injected event;
a detection there that matches nothing is an induced false alarm.

teachers -> results/student/injection/
    teacher_matches.csv    one row per injected event x teacher: hit, snr_db, duration, freq, IoU, onset/offset error
    teacher_recall.csv     recall per teacher x SNR (95% CI: cluster bootstrap over nights, 2000 resamples),
                           the mean over teachers, the share of events no teacher found, and >= 2 of 4
    identifiability.json   lowest SNR from which the mean teacher recall stays > 0.5 (the condition of the
                           trace theorem, Tanno 2019 / Zhang 2020), with its bootstrap CI at each SNR
    teacher_false_alarms.csv   induced false alarms per quiet N2 minute
Students are scored with score_events() by selection.py (6.2).
"""
import argparse
import json

import numpy as np
import pandas as pd

import data
from spindlebench import provenance
from spindlebench.metrics.events import iou_matrix, match

OUT = data.STUDY / "injection"
TOL = 0.5
BOOT = 2000


def load_truth(v, ds, rec):
    ev = pd.read_csv(OUT / f"v{v}" / "events" / ds / f"{rec}.csv")
    q = np.load(OUT / f"v{v}" / "quiet" / ds / f"{rec}.npy")
    return ev, q


def scored_region(ev, quiet_bits, n):
    """quiet N2 or inside an injected event, on the 100 Hz grid"""
    region = np.unpackbits(quiet_bits)[:n].astype(bool)
    for o, d in zip(ev.onset, ev.duration):
        region[int(round(o * data.FS)):int(round((o + d) * data.FS))] = True
    return region


def score_events(det, ev, region):
    """det [n, >=2] onset/offset s; ev injected events -> (per-event DataFrame, n false alarms, n considered)
    per event: hit, iou (best-overlapping detection), onset_err, offset_err (matched detection minus truth),
    n_overlapping (detections overlapping the event: fragmentation)"""
    det = np.asarray(det, float).reshape(-1, det.shape[1] if len(det) else 2)[:, :2]
    c = det.mean(1) if len(det) else np.zeros(0)
    i = np.round(c * data.FS).astype(int)
    keep = (i >= 0) & (i < len(region))
    keep[keep] = region[i[keep]]
    det, c = det[keep], c[keep]
    tr = np.c_[ev.onset, ev.onset + ev.duration]
    tc = tr.mean(1)
    hit = np.zeros(len(tr), bool)
    mdet = np.full(len(tr), -1)
    # exact one-to-one matching by centre, per cluster of points closer than TOL
    if len(det) and len(tr):
        x = np.r_[c, tc]
        lab = np.r_[np.zeros(len(c), int), np.ones(len(tc), int)]
        idx = np.r_[np.arange(len(c)), np.arange(len(tc))]
        o = np.argsort(x, kind="stable")
        x, lab, idx = x[o], lab[o], idx[o]
        cut = np.r_[0, np.flatnonzero(np.diff(x) > TOL + 1e-9) + 1, len(x)]
        for a, b in zip(cut[:-1], cut[1:]):
            dd, tt = idx[a:b][lab[a:b] == 0], idx[a:b][lab[a:b] == 1]
            if len(dd) and len(tt):
                _, _, _, pairs = match(np.c_[c[dd], 0 * dd], np.c_[tc[tt], 0 * tt], "onset", TOL)
                for pi, pj in pairs:
                    hit[tt[pj]] = True
                    mdet[tt[pj]] = dd[pi]
    rows = pd.DataFrame(dict(hit=hit))
    if len(det) and len(tr):
        iou = iou_matrix(np.c_[det[:, 0], det[:, 1] - det[:, 0]], np.c_[tr[:, 0], tr[:, 1] - tr[:, 0]])
        rows["iou"] = iou.max(0)
        rows["n_overlapping"] = (iou > 0).sum(0)
    else:
        rows["iou"] = 0.0
        rows["n_overlapping"] = 0
    m = mdet >= 0
    rows["onset_err"] = np.where(m, det[np.maximum(mdet, 0), 0] - tr[:, 0], np.nan) if len(det) else np.nan
    rows["offset_err"] = np.where(m, det[np.maximum(mdet, 0), 1] - tr[:, 1], np.nan) if len(det) else np.nan
    fa = int(len(det) - hit.sum())
    return rows, fa, len(det)


def boot_ci(df, value, by, rng, n=BOOT):
    """cluster bootstrap over nights of the mean of `value` within each `by` group -> {group: (lo, hi)}"""
    nights = df.night.unique()
    g = df.groupby(["night", by])[value].agg(["sum", "count"]).unstack(by).fillna(0)
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    draws = rng.integers(0, len(nights), (n, len(nights)))
    est = s[draws].sum(1) / np.maximum(c[draws].sum(1), 1)
    lo, hi = np.percentile(est, [2.5, 97.5], axis=0)
    return dict(zip(g["sum"].columns, zip(lo, hi)))


def cmd_teachers(variants):
    provenance.track(__file__, protocols=["artefacts.v1"])
    man = data.manifest().set_index(["dataset", "rec"])
    sp = data.splits()
    val = sp[sp.role == "val"]
    rows, fas = [], []
    for v in variants:
        store = OUT / f"v{v}" / "detections"
        for ds, rec in zip(val.dataset, val.rec):
            ev, q = load_truth(v, ds, rec)
            n = int(np.ceil(man.loc[(ds, rec)].duration * data.FS))
            region = scored_region(ev, q, n)
            quiet_min = np.unpackbits(q)[:n].sum() / data.FS / 60
            for t in data.TEACHERS:
                f = store / ds / t / f"{rec}.csv"
                d = pd.read_csv(f)
                d = d[d.event == "spindle"]
                det = np.c_[d.onset, d.onset + d.duration.fillna(0)]
                r, fa, _ = score_events(det, ev, region)
                r = pd.concat([ev[["onset", "duration", "freq", "snr_db"]].reset_index(drop=True), r], axis=1)
                r.insert(0, "teacher", t)
                r.insert(0, "night", f"{ds}/{rec}")
                r.insert(0, "variant", v)
                rows.append(r)
                fas.append(dict(variant=v, night=f"{ds}/{rec}", teacher=t, false_alarms=fa, quiet_min=quiet_min))
    mt = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    mt.to_csv(OUT / "teacher_matches.csv", index=False)
    rng = np.random.default_rng(data.SEED)
    out = []
    for t, g in mt.groupby("teacher"):
        ci = boot_ci(g.assign(hit=g.hit.astype(float)), "hit", "snr_db", rng)
        for snr, h in g.groupby("snr_db").hit:
            out.append(dict(teacher=t, snr_db=snr, recall=h.mean(), lo=ci[snr][0], hi=ci[snr][1], events=len(h)))
    # per event: how many of the four teachers found it
    k = mt.pivot_table(index=["variant", "night", "onset"], columns="teacher", values="hit", aggfunc="first")
    ev = mt.drop_duplicates(["variant", "night", "onset"]).set_index(["variant", "night", "onset"])
    k = k.join(ev[["snr_db"]])
    k["n_found"] = k[list(data.TEACHERS)].sum(1)
    k["mean_recall"] = k[list(data.TEACHERS)].mean(1)
    k = k.reset_index()
    ci_mean = boot_ci(k, "mean_recall", "snr_db", rng)
    for snr, g in k.groupby("snr_db"):
        out.append(dict(teacher="mean of 4", snr_db=snr, recall=g.mean_recall.mean(), lo=ci_mean[snr][0],
                        hi=ci_mean[snr][1], events=len(g)))
        out.append(dict(teacher="none found", snr_db=snr, recall=(g.n_found == 0).mean(), events=len(g)))
        out.append(dict(teacher=">= 2 of 4", snr_db=snr, recall=(g.n_found >= 2).mean(), events=len(g)))
    rec = pd.DataFrame(out)
    rec.to_csv(OUT / "teacher_recall.csv", index=False)
    m = rec[rec.teacher == "mean of 4"].sort_values("snr_db")
    above = (m.recall > 0.5).to_numpy()
    ok = [s for i, s in enumerate(m.snr_db) if above[i:].all()]
    ok_lo = [s for i, s in enumerate(m.snr_db) if (m.lo.to_numpy()[i:] > 0.5).all()]
    ident = dict(snr_from=min(ok) if ok else None, snr_from_ci_lower_bound=min(ok_lo) if ok_lo else None,
                 mean_recall=dict(zip(m.snr_db, m.recall.round(4))), variants=list(variants),
                 events=int(len(k)), note="mean over the four teachers of P(y_r = 1 | y = 1) on injected spindles")
    (OUT / "identifiability.json").write_text(json.dumps(ident, indent=1))
    fa = pd.DataFrame(fas)
    fa_s = fa.groupby("teacher")[["false_alarms", "quiet_min"]].sum()
    fa_s["per_quiet_min"] = fa_s.false_alarms / fa_s.quiet_min
    fa_s.to_csv(OUT / "teacher_false_alarms.csv")
    pd.set_option("display.width", 200)
    print(rec.pivot_table(index="teacher", columns="snr_db", values="recall").round(3).to_string())
    print(json.dumps(ident, indent=1))
    print(fa_s.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["teachers"])
    ap.add_argument("--variants", default="0,1,2,3")
    a = ap.parse_args()
    cmd_teachers([int(v) for v in a.variants.split(",")])


if __name__ == "__main__":
    main()
