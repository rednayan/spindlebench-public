"""
Teacher events -> 100 Hz binary masks, and the corpus agreement tables (plan 3.2):

    uv run python rasterise.py [--workers 10]       (from experiments/student; after prepare.py)

Per training-site recording -> results/student/work/teachers/<dataset>/<rec>.npz (compressed):
    a7, luna_spindles, wonambi_ucsd, yasa_default   uint8 [n], 1 inside the teacher's events
    valid                                           uint8 [n], 1 = N2 (public hypnogram) and clean
n = the length of the prepared 100 Hz signal. Teachers come from results/detections_clean (run on
artefact-free windows, artefacts.v1); "clean" also leaves out JOIN_GUARD = 1 s on each side of a
flagged window, where the teachers dropped their events. Scores are ignored (binary annotators).

Tables -> results/student/teachers/ (computed on valid samples / events whose centre is valid):
    per_recording.csv  dataset, rec, role, teacher, valid_min, events, per_min, W/REM/N3 densities
                       (clean, for the plausibility ratios), median duration
    density.csv        median over recordings per dataset x teacher
    agreement.csv      pairwise event F1 (centres within ±0.5 s, one-to-one) pooled per dataset,
                       and sample-level Cohen's kappa on valid samples
    patterns.csv       dataset, rec, role, pattern (bit i = TEACHERS[i]), count of valid samples;
                       the sufficient statistic for every label model in label_models.py
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd

import data
from spindlebench import provenance
from spindlebench.metrics.events import match, prf

OUT = data.STUDY / "teachers"
TOL = 0.5
BITS = 1 << np.arange(len(data.TEACHERS))


def match_tp(a, b, tol=TOL):
    """exact maximum one-to-one matching of centres within tol (metrics.events.match), solved per
    cluster of points separated by gaps > tol (clusters cannot share pairs, so the sum is exact)"""
    if not len(a) or not len(b):
        return 0
    x = np.r_[a, b]
    lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    o = np.argsort(x, kind="stable")
    x, lab = x[o], lab[o]
    cut = np.r_[0, np.flatnonzero(np.diff(x) > tol + 1e-9) + 1, len(x)]
    tp = 0
    for i, j in zip(cut[:-1], cut[1:]):
        xa, xb = x[i:j][lab[i:j] == 0], x[i:j][lab[i:j] == 1]
        if len(xa) and len(xb):
            tp += match(np.c_[xa, 0 * xa], np.c_[xb, 0 * xb], "onset", tol)[0]
    return tp


def patterns(Y, valid):
    """counts of the 2^R teacher patterns over valid samples"""
    code = (Y[:, valid].astype(np.int64) * BITS[:, None]).sum(0)
    return np.bincount(code, minlength=1 << len(data.TEACHERS))


def kappa_counts(a, b):
    """2x2 table [[n00, n01], [n10, n11]] of two binary vectors"""
    return np.bincount(2 * a.astype(np.int64) + b, minlength=4).reshape(2, 2)


def one(args):
    row, role = args
    data.guard(row.dataset)
    n = len(np.load(data.prep_path(row.dataset, row.rec))["x"])
    stage = data.stage_per_sample(row.dataset, row.rec, n)
    clean = data.clean_per_sample(row.dataset, row.rec, n)
    valid = (stage == "N2") & clean
    Y, cen, recs = {}, {}, []
    for t in data.TEACHERS:
        on, du = data.load_teacher_events(row.dataset, row.rec, t, row.default_channel)
        Y[t] = data.events_to_mask(on, du, n)
        c = np.round((on + np.nan_to_num(du) / 2) * data.FS).astype(int)
        inside = (c >= 0) & (c < n)
        c, du = c[inside], du[inside]
        st, ok = stage[c], clean[c]
        sel = ok & (st == "N2")
        r = dict(dataset=row.dataset, rec=row.rec, role=role, teacher=t, valid_min=valid.sum() / data.FS / 60)
        r["events"] = int(sel.sum())
        r["dur_median"] = float(np.median(du[sel])) if sel.any() else np.nan
        for s in ("N2", "N3", "W", "REM"):
            mins = ((stage == s) & clean).sum() / data.FS / 60
            r[f"{s}_per_min"] = (ok & (st == s)).sum() / mins if mins else np.nan
        recs.append(r)
        cen[t] = np.sort(c[sel] / data.FS)
    out = data.teacher_path(row.dataset, row.rec)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, valid=valid.astype(np.uint8), **Y)
    Ym = np.stack([Y[t] for t in data.TEACHERS])
    pat = patterns(Ym, valid)
    pairs = {}
    for i, j in combinations(range(len(data.TEACHERS)), 2):
        a, b = data.TEACHERS[i], data.TEACHERS[j]
        tp = match_tp(cen[a], cen[b])
        pairs[(a, b)] = (tp, len(cen[a]) - tp, len(cen[b]) - tp, kappa_counts(Ym[i, valid], Ym[j, valid]))
    return recs, pat, pairs


def kappa(t):
    n = t.sum()
    po = (t[0, 0] + t[1, 1]) / n
    pe = (t[0].sum() * t[:, 0].sum() + t[1].sum() * t[:, 1].sum()) / n ** 2
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    provenance.track(__file__, protocols=["artefacts.v1"])
    m = data.manifest()
    role = data.splits().set_index(["dataset", "rec"]).role
    jobs = [(r, role[(r.dataset, r.rec)]) for _, r in m.iterrows()]
    recs, pats, pair_sum = [], [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for k, ((rr, pat, pairs), (row, rl)) in enumerate(zip(ex.map(one, jobs, chunksize=2), jobs)):
            recs += rr
            pats += [dict(dataset=row.dataset, rec=row.rec, role=rl, pattern=p, count=int(c))
                     for p, c in enumerate(pat)]
            for key, (tp, fp, fn, tab) in pairs.items():
                s = pair_sum.setdefault((row.dataset, *key), [0, 0, 0, np.zeros((2, 2), np.int64)])
                s[0] += tp; s[1] += fp; s[2] += fn; s[3] = s[3] + tab
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                print(f"{k + 1}/{len(jobs)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(recs)
    per.to_csv(OUT / "per_recording.csv", index=False)
    pd.DataFrame(pats).to_csv(OUT / "patterns.csv", index=False)
    g = per.groupby(["dataset", "teacher"])
    dens = pd.DataFrame({"N2_per_min": g.N2_per_min.median(), "N3_per_min": g.N3_per_min.median(),
                         "W_over_N2": g.W_per_min.median() / g.N2_per_min.median(),
                         "REM_over_N2": g.REM_per_min.median() / g.N2_per_min.median(),
                         "dur_median_s": g.dur_median.median(),
                         "nights_below_0.2_per_min": g.N2_per_min.apply(lambda v: (v < 0.2).mean()),
                         "events_total": g.events.sum()}).round(3)
    dens.to_csv(OUT / "density.csv")
    agr = pd.DataFrame([dict(dataset=ds, a=x, b=y, tp=tp, fp=fp, fn=fn, f1=prf(tp, fp, fn)[2], kappa=kappa(tab))
                        for (ds, x, y), (tp, fp, fn, tab) in pair_sum.items()]).round(4)
    agr.to_csv(OUT / "agreement.csv", index=False)
    print(dens.to_string())
    print(agr.pivot_table(index=["a", "b"], columns="dataset", values=["f1", "kappa"]).round(2).to_string())
    print(f"wrote {OUT}/ and {data.WORK / 'teachers'}/")


if __name__ == "__main__":
    main()
