"""
Label models: the training targets and the "consensus as detector" rows (plan 3.3, 3.4).

    uv run python label_models.py fit        targets from the training nights (after rasterise.py)
    uv run python label_models.py gold       consensus detections on DREAMS and NAP
    (from experiments/student; then score the gold rows with experiments/rq1_benchmark/evaluate.py,
     see README.md)

With R = 4 binary teachers and one global confusion matrix per teacher, every label model is a
function of the teacher pattern at a sample (2^4 = 16 patterns). Each model is therefore stored as a
16-entry posterior table, exact for every sample; data.py looks targets up from the stored teacher
masks instead of storing a posterior array per night.

    mv     1 if >= 2 of 4 teachers (agreement >= 0.5, Warby et al. 2014)
    soft   fraction of teachers, k/4
    ds     Dawid-Skene posterior P(y=1 | pattern), crowd-kit 1.4.2 DawidSkene(n_iter=100, tol=-inf;
           majority-vote initialisation), binary, 4 workers. tol=-inf: crowd-kit stops when its
           lower bound does not rise by tol, and on these data that bound falls after the 2nd
           iteration, which stopped EM far from convergence (P(spindle | only Luna) 0.25 vs 0.46).
           With the stop disabled it equals an exact EM on the pattern counts (ds_exact, converged
           in 68 iterations) to 1e-4; that is asserted on every fit.
    fs     FlyingSquid triplet posterior (Fu et al. 2020), flyingsquid 0.0.0a0, LabelModel(4)
           .fit(L in {-1,+1}, class_balance=[1-pi, pi]) with pi = mean soft target (no labels).
           [ours] the class balance; the package needs one and has no label-free estimate.
Fitting data: valid samples (N2, clean) of the training nights (role == train in splits.csv), all of
them summarised by their pattern counts (results/student/teachers/patterns.csv); DS and FS are fitted
on 2,000,000 samples drawn from those counts (multinomial, seed 2026; = a uniform random sample of
training samples). Per-site fits (DS by ds_exact) are written as a diagnostic only.

fit  -> results/student/work/targets/label_models.json and results/student/teachers/label_models.csv
gold -> results/student/detections/<dataset>/consensus_<model>[_corpus]/<rec>.csv, the detection
        file contract. Teachers = their artefact-free runs (results/detections_clean). Per sample:
        posterior >= 0.5 -> runs -> SUMOv2's merge_and_filter_spindles (decoder D0). Two variants:
        consensus_<m>         label model fitted on that gold set's own teacher outputs (all clean
                              samples, stage-free: no expert staging, no expert labels), plan 3.4
        consensus_<m>_corpus  the training-corpus table (the target the student actually learns)
        No expert label is read here; evaluate.py scores the files.
"""
import argparse
import json

import numpy as np
import pandas as pd

import data
from spindlebench import provenance

R = len(data.TEACHERS)
P = 1 << R
BITS = 1 << np.arange(R)
PAT = ((np.arange(P)[:, None] & BITS) > 0).astype(int)  # [16, R], PAT[p, i] = teacher i says 1
VOTES = PAT.sum(1)
N_FIT = 2_000_000
DS_ITER = 100
TARGETS = data.WORK / "targets" / "label_models.json"
GOLD_OUT = data.STUDY / "detections"


def sample_patterns(counts, n=N_FIT, seed=data.SEED):
    """pattern ids of n samples drawn from the pattern counts (with replacement)"""
    counts = np.asarray(counts, float)
    k = np.random.default_rng(seed).multinomial(n, counts / counts.sum())
    return np.repeat(np.arange(P), k)


def fit_ds(pat_ids):
    from crowdkit.aggregation import DawidSkene
    n = len(pat_ids)
    df = pd.DataFrame({"task": np.repeat(np.arange(n), R), "worker": np.tile(np.array(data.TEACHERS), n),
                       "label": PAT[pat_ids].ravel()})
    ds = DawidSkene(n_iter=DS_ITER, tol=-np.inf).fit(df)
    prior = ds.priors_.reindex([0, 1]).to_numpy(float)
    err = {t: ds.errors_.loc[t].reindex(index=[0, 1], columns=[0, 1]).to_numpy(float) for t in data.TEACHERS}
    # the table is crowd-kit's own posterior (probas_, its last E-step) for each pattern; a pattern
    # absent from the fitted sample gets the E-step formula on the final parameters
    like = np.ones((P, 2)) * prior
    for i, t in enumerate(data.TEACHERS):
        like *= err[t][PAT[:, i], :]  # errors_[observed, true]
    post = like[:, 1] / like.sum(1)
    seen = ds.probas_.reindex(columns=[0, 1])[1].to_numpy()
    first = pd.Series(np.arange(n)).groupby(pat_ids).first()
    post[first.index] = seen[first.to_numpy()]
    exact, _ = ds_exact(np.bincount(pat_ids, minlength=P))
    assert np.allclose(post, exact, atol=1e-4), "crowd-kit DS != exact EM on the same samples"
    conf = {t: dict(recall=float(err[t][1, 1]), specificity=float(err[t][0, 0])) for t in data.TEACHERS}
    return post, dict(prior_spindle=float(prior[1]), teachers=conf, n_iter=DS_ITER, tol="-inf")


def ds_exact(counts, tol=1e-12, max_iter=100_000):
    """Dawid & Skene (1979) EM, binary, one 2x2 matrix per teacher, on the pattern counts: the same
    model as crowd-kit's, on all samples, run to convergence. A check on fit_ds, not a target."""
    w = np.asarray(counts, float)
    q = (VOTES >= 2).astype(float)  # majority-vote initialisation, as crowd-kit
    for k in range(max_iter):
        w1, w0 = w * q, w * (1 - q)
        pi = w1.sum() / w.sum()
        sens = (w1[:, None] * PAT).sum(0) / w1.sum()
        spec = (w0[:, None] * (1 - PAT)).sum(0) / w0.sum()
        l1 = pi * np.prod(np.where(PAT, sens, 1 - sens), 1)
        l0 = (1 - pi) * np.prod(np.where(PAT, 1 - spec, spec), 1)
        q_new = l1 / (l1 + l0)
        if np.abs(q_new - q).max() < tol:
            return q_new, dict(prior_spindle=float(pi), recall=sens.tolist(), specificity=spec.tolist(), iterations=k)
        q = q_new
    raise RuntimeError("DS EM did not converge")


def fit_fs(pat_ids, pi):
    if not hasattr(np, "product"):  # flyingsquid 0.0.0a0 / pgmpy 0.1.12 call np.product (removed in numpy 2)
        np.product = np.prod
    from flyingsquid.label_model import LabelModel
    L = 2 * PAT[pat_ids] - 1
    lm = LabelModel(R)
    lm.fit(L, class_balance=np.array([1 - pi, pi]))
    post = lm.predict_proba(2 * PAT - 1)[:, 1]
    acc = [float(a) for a in np.ravel(lm.estimated_accuracies())]
    return post, dict(class_balance=float(pi), accuracies=dict(zip(data.TEACHERS, acc)))


def fit_all(counts, exact_ds=False):
    """all four tables; exact_ds uses ds_exact on the full counts (per-site diagnostics only)"""
    pid = sample_patterns(counts)
    soft = VOTES / R
    pi = float((counts * soft).sum() / counts.sum())
    if exact_ds:
        ds, ds_info = ds_exact(counts)
        ds_info["teachers"] = {t: dict(recall=r, specificity=sp) for t, r, sp in
                               zip(data.TEACHERS, ds_info.pop("recall"), ds_info.pop("specificity"))}
    else:
        ds, ds_info = fit_ds(pid)
    fs, fs_info = fit_fs(pid, pi)
    return dict(mv=(VOTES >= 2).astype(float), soft=soft, ds=ds, fs=fs), dict(ds=ds_info, fs=fs_info,
                                                                               n_fit=len(pid), mean_soft=pi)


def table(models, counts):
    t = pd.DataFrame(PAT, columns=list(data.TEACHERS))
    t.insert(0, "pattern", np.arange(P))
    t["votes"] = VOTES
    t["share_of_valid_samples"] = counts / counts.sum()
    for k, v in models.items():
        t[k] = np.round(v, 4)
    return t


def cmd_fit():
    provenance.track(__file__, protocols=["artefacts.v1"])
    pat = pd.read_csv(data.STUDY / "teachers" / "patterns.csv")
    pat = pat[pat.dataset.isin(data.SITES)]  # (patterns.csv only ever holds training sites)
    tr = pat[pat.role == "train"]
    counts = tr.groupby("pattern")["count"].sum().reindex(range(P), fill_value=0).to_numpy()
    models, info = fit_all(counts)
    diag = {}
    for site in data.SITES:
        c = tr[tr.dataset == site].groupby("pattern")["count"].sum().reindex(range(P), fill_value=0).to_numpy()
        m_s, i_s = fit_all(c, exact_ds=True)
        diag[site] = dict(ds=m_s["ds"].tolist(), fs=m_s["fs"].tolist(), info=i_s)
    out = dict(teachers=list(data.TEACHERS), bit_order="pattern = sum_i 2^i * y_i, i over teachers",
               fitted_on=dict(role="train", sites=list(data.SITES), valid_samples=int(counts.sum()),
                              nights=int(tr.rec.nunique())),
               models={k: v.tolist() for k, v in models.items()}, info=info, per_site_diagnostic=diag)
    TARGETS.parent.mkdir(parents=True, exist_ok=True)
    TARGETS.write_text(json.dumps(out, indent=1))
    t = table(models, counts)
    t.to_csv(data.STUDY / "teachers" / "label_models.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    print(json.dumps(info, indent=1))
    for site, d in diag.items():
        print(site, "DS recall", {k: round(v["recall"], 3) for k, v in d["info"]["ds"]["teachers"].items()},
              "prior", round(d["info"]["ds"]["prior_spindle"], 4))


def load_targets():
    j = json.loads(TARGETS.read_text())
    assert tuple(j["teachers"]) == data.TEACHERS
    return {k: np.asarray(v) for k, v in j["models"].items()}


def ds_confusion():
    """the fitted Dawid-Skene confusion matrix of every teacher, [R, 2, 2] in data.TEACHERS order,
    rows = true class (background, spindle), columns = the teacher's label: [[spec, 1 - spec], [1 - rec, rec]]"""
    j = json.loads(TARGETS.read_text())
    assert tuple(j["teachers"]) == data.TEACHERS
    t = j["info"]["ds"]["teachers"]
    return np.array([[[t[r]["specificity"], 1 - t[r]["specificity"]], [1 - t[r]["recall"], t[r]["recall"]]]
                     for r in data.TEACHERS])


# ------------------------------------------------------------------ gold sets (3.4)

def gold_recording(row):
    """teacher pattern per 100 Hz sample and the clean mask (no staging, no expert labels)"""
    n = int(np.ceil(row.duration * data.FS))
    Y = np.stack([data.events_to_mask(*data.load_teacher_events(row.dataset, row.rec, t, row.default_channel), n)
                  for t in data.TEACHERS])
    code = (Y.astype(np.int64) * BITS[:, None]).sum(0)
    return code, data.clean_per_sample(row.dataset, row.rec, n)


def decode(post, code):
    """posterior >= 0.5 -> runs -> D0 (SUMOv2 merge_and_filter_spindles) -> rows of the contract"""
    p = post[code]
    ev = data.mask_to_events(p >= 0.5) / data.FS
    if not len(ev):
        return ev, p
    return data.sumo().merge_and_filter_spindles(ev.astype(float)), p


def write_detections(ds, rec, det_id, channel, ev, p):
    rows = [dict(onset=round(s, 3), duration=round(e - s, 3), channel=channel, event="spindle",
                 score=round(float(p[int(round(s * data.FS)):int(round(e * data.FS))].mean()), 4)) for s, e in ev]
    f = GOLD_OUT / ds / det_id / f"{rec}.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["onset", "duration", "channel", "event", "score"]).to_csv(f, index=False)


def cmd_gold():
    provenance.track(__file__, protocols=["artefacts.v1"])
    corpus = load_targets()
    m = pd.read_csv(data.paths.CACHE / "manifest.csv")
    fits = {}
    for ds in data.GOLD:
        rows = [r for _, r in m[m.dataset == ds].iterrows()]
        recs = {r.rec: gold_recording(r) for r in rows}
        counts = np.zeros(P)
        for code, clean in recs.values():
            counts += np.bincount(code[clean], minlength=P)
        own, info = fit_all(counts)
        fits[ds] = dict(models={k: v.tolist() for k, v in own.items()}, info=info,
                        clean_samples=int(counts.sum()))
        n_ev = {}
        for r in rows:
            code, _ = recs[r.rec]
            for name, tab in [(k, own[k]) for k in ("mv", "ds", "fs")] + \
                             [(f"{k}_corpus", corpus[k]) for k in ("ds", "fs")]:
                ev, p = decode(tab, code)
                write_detections(ds, r.rec, f"consensus_{name}", str(r.default_channel), ev, p)
                n_ev[name] = n_ev.get(name, 0) + len(ev)
        print(ds, "events:", n_ev)
        print(table(own, counts)[["pattern", *data.TEACHERS, "votes", "share_of_valid_samples", "ds", "fs"]]
              .to_string(index=False))
    (data.STUDY / "consensus").mkdir(parents=True, exist_ok=True)
    (data.STUDY / "consensus" / "gold_label_models.json").write_text(json.dumps(fits, indent=1))
    print(f"wrote {GOLD_OUT}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "gold"])
    a = ap.parse_args()
    {"fit": cmd_fit, "gold": cmd_gold}[a.cmd]()


if __name__ == "__main__":
    main()
