"""
Report of the GPU-track student grid (plan notes/plans/student_architecture.md, sections 8 and 8.6).
Reads only stored outputs: run folders, selection, gold-set detections and p(t) traces. Trains nothing.

    uv run python experiments/student/report.py          (from the repo root, root environment)

A. Training and selection on validation nights (no expert): per run and per arm (5 seeds).
B. DREAMS, benchmark.v1 rows already computed by evaluate.py: expert 1, N2, onset +-0.5 s (primary),
   plus onset +-0.25 s, IoU >= 0.2 and expert 2 onset +-0.5 s. Pooled F1 per run; arm = mean over seeds.
C. NAP, nap_points.v1 (primary: integrated F1 over centre tolerances 0.25-1.5 s, N2, C3), computed here
   for every student run, the consensus rows, the four teachers as the students saw them
   (results/detections_clean) and the open baselines (results/detections). Person-level bootstrap.
D. Hypotheses H1-H3 and H5 (H4 in E) for the headline arm student_rules4_fixA_gpu, seeds averaged:
   DREAMS pooled-F1 difference with a recording bootstrap (2000, seed 0) and the exact Wilcoxon
   signed-rank test on per-recording F1 (n = 8, smallest two-sided p 0.0078); NAP integrated-F1
   difference with a paired person bootstrap and Wilcoxon on per-person integrated F1 (17 people);
   H5 also with the Bayesian signed-rank test (Benavoli et al. 2014, prior strength 0.5 at 0, ROPE
   +-0.02) on per-recording F1. Secondary contrasts (ablations, student vs its own target) are
   reported the same way and are not tests.
E. H4: on DREAMS excerpts 1-6, first 1000 s (both experts), N2, the expert events are split into
   "both experts" (one-to-one onset +-0.5 s) and "one expert only". Uncertainty of the 5-seed ensemble
   over each event = mean over the event of H(mean p) + mean over seeds of KL(p_s || mean p).
   AUROC for "one expert only"; pre-registered threshold 0.7. Excerpt bootstrap interval.
-> results/student/report/{runs.csv, arms.csv, dreams.csv, nap.csv, nap_per_person.csv,
                           hypotheses.csv, contrasts.csv, h4.csv, report.md}
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from spindlebench import paths, provenance
from spindlebench.data.nap import load_det, nap_info, naps
from spindlebench.metrics import events as em
from spindlebench.protocols import nap_points
from spindlebench.protocols.nap_points import KEYS, TAUS, integrated, score_nap, summarise, tensors

provenance.track(__file__, protocols=[nap_points.VERSION])

RES = paths.RESULTS
ST = RES / "student"
OUT = ST / "report"
RUNS = Path(os.environ.get("SPINDLEBENCH_RUNS") or paths.ROOT.parent / "runs" / "student").resolve()
HEAD = "student_rules4_fixA_gpu"
B = 2000
ROPE = 0.02

ARMS = {  # arm -> (short label, group)
    "student_rules4_fixA_gpu": ("4 teachers, noise model anchored at DS (headline)", "teacher-aware"),
    "student_rules4_gpu": ("4 teachers, free noise model, lambda 0.01", "teacher-aware"),
    "student_rules4_lam0_gpu": ("4 teachers, free noise model, lambda 0", "ablation"),
    "student_rules4_global_gpu": ("4 teachers, one matrix per teacher", "ablation"),
    "student_rules4_noB_gpu": ("4 teachers, no stream B", "ablation"),
    "student_rules4_nohead2_gpu": ("4 teachers, no boundary head", "ablation"),
    "student_soft_gpu": ("soft targets k/4", "plain student"),
    "student_mv_gpu": ("majority vote targets", "plain student"),
    "student_ds_gpu": ("Dawid-Skene targets", "plain student"),
    "student_a7_gpu": ("one teacher: A7", "single teacher"),
    "student_luna_gpu": ("one teacher: Luna", "single teacher"),
    "student_ucsd_gpu": ("one teacher: Wonambi UCSD", "single teacher"),
    "student_yasa_gpu": ("one teacher: YASA", "single teacher"),
}
TEACHERS = {"a7": "A7", "luna_spindles": "Luna", "wonambi_ucsd": "Wonambi UCSD", "yasa_default": "YASA default"}
CONSENSUS = {"consensus_mv": "teachers' majority vote", "consensus_ds": "teachers' Dawid-Skene posterior",
             "consensus_fs": "teachers' FlyingSquid posterior", "consensus_ds_corpus": "Dawid-Skene fitted on corpus",
             "consensus_fs_corpus": "FlyingSquid fitted on corpus"}
OPEN = {"sumov2": "SUMOv2 (expert-trained)", "sumo": "SUMO (expert-trained)",
        "wonambi_moelle2011": "Moelle 2011 (rule)"}
TEACHER_OF = {"student_a7_gpu": "a7", "student_luna_gpu": "luna_spindles", "student_ucsd_gpu": "wonambi_ucsd",
              "student_yasa_gpu": "yasa_default"}
TARGET_OF = {"student_mv_gpu": "consensus_mv", "student_ds_gpu": "consensus_ds"}

DREAMS_RULES = [("E1 onset 0.5 (primary)", "expert1", "onset", 0.5), ("E1 onset 0.25", "expert1", "onset", 0.25),
                ("E1 IoU 0.2", "expert1", "iou", 0.2), ("E2 onset 0.5", "expert2", "onset", 0.5)]


def arm_of(run):
    return re.sub(r"_s\d+$", "", run)


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), np.nan)


# ------------------------------------------------------------------ A. training and selection

def section_a():
    sel = pd.read_csv(ST / "selection" / "summary_gpu.csv")
    rows = []
    for r in sel.itertuples():
        d = RUNS / r.run
        done = json.loads((d / "done.json").read_text())
        met = pd.read_csv(d / "metrics.csv").set_index("pass")
        m = met.loc[done["best_pass"]]
        tim = json.loads((d / "timing.json").read_text())
        rows.append(dict(run=r.run, arm=arm_of(r.run), seed=int(r.run.rsplit("_s", 1)[1]),
                         best_pass=done["best_pass"], passes_run=done["passes_run"],
                         hours=(sum(tim.get("pass_s", [])) + sum(tim.get("val_s", []))) / 3600,
                         val_f1_mv=r.f1_mv, val_f1_ds=m.get("f1_ds"), val_f1_a7=m.get("f1_a7"),
                         inj_f1=r.inj_f1, inj_precision=r.precision, n2_per_min=r.per_min_N2,
                         dur_median=m.get("dur_median"), wake_over_n2=m.get("W_over_N2"),
                         rem_over_n2=m.get("REM_over_N2"), diag0=m.get("train_diag0"), diag1=m.get("train_diag1"),
                         excluded=r.excluded))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B. DREAMS

def dreams_tables():
    s = pd.concat([pd.read_csv(ST / "benchmark" / "summary_partial.csv"),
                   pd.read_csv(ST / "consensus_teachers" / "summary_partial.csv"),
                   pd.read_csv(RES / "benchmark" / "summary.csv").query("detector in @OPEN")])
    p = pd.concat([pd.read_csv(ST / "benchmark" / "per_recording_partial.csv"),
                   pd.read_csv(ST / "consensus_teachers" / "per_recording_partial.csv"),
                   pd.read_csv(RES / "benchmark" / "per_recording.csv").query("detector in @OPEN")])
    s = s[(s.dataset == "dreams_spindles") & (s.scope == "N2")]
    p = p[(p.dataset == "dreams_spindles") & (p.scope == "N2") & ~p.excluded.astype(bool)]
    return s, p


def pick(df, ref, rule, tol):
    return df[(df.reference == ref) & (df.rule == rule) & np.isclose(df.tol, tol)]


# ------------------------------------------------------------------ C. NAP (nap_points.v1)

def nap_scores(dets):
    """dets: {name: store dir} -> per-person counts C (n_subj, keys, 3), aff, subjects"""
    infos = [nap_info(r) for r in naps().itertuples()]
    subjects = sorted({i["subject"] for i in infos})
    out = {}
    for name, store in dets.items():
        rows = []
        for info in infos:
            on, off = load_det(store / f"{info['rec']}.csv", info)
            rows.append((info["subject"], score_nap(on, off, info["marks"], info["T"])))
        out[name] = tensors(rows, subjects)
    return out, subjects


# ------------------------------------------------------------------ statistics

def boot_idx(n, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def bayes_signed_rank(z, rope=ROPE, s=0.5, n=20000, seed=0):
    """Benavoli et al. 2014: P(left), P(rope), P(right) for the differences z (A - B)"""
    rng = np.random.default_rng(seed)
    zz = np.r_[0.0, z]
    w = rng.dirichlet(np.r_[s, np.ones(len(z))], size=n)
    S = zz[:, None] + zz[None, :]
    L, R_ = (S < -2 * rope), (S > 2 * rope)
    M = ~(L | R_)
    th = np.stack([np.einsum("ni,ij,nj->n", w, X.astype(float), w) for X in (L, M, R_)], 1)
    win = th.argmax(1)
    return [(win == k).mean() for k in range(3)]


def dreams_pair(p, a_names, b_names, ref="expert1", rule="onset", tol=0.5):
    """pooled F1 (seed-averaged) difference A - B over recordings, bootstrap CI, Wilcoxon on per-recording F1"""
    q = pick(p, ref, rule, tol)
    recs = sorted(q.rec.unique())

    def counts(names):
        c = np.stack([q[q.detector == n].set_index("rec").loc[recs, ["tp", "fp", "fn"]].to_numpy(float) for n in names])
        return c  # (n_names, n_rec, 3)

    A, Bc = counts(a_names), counts(b_names)
    idx = boot_idx(len(recs))

    def pooled(c, ix=None):
        t = c.sum(1) if ix is None else c[:, ix].sum(2)
        return f1(t[..., 0], t[..., 1], t[..., 2]).mean(0)

    d0 = pooled(A) - pooled(Bc)
    db = pooled(A, idx) - pooled(Bc, idx)
    fa = np.nanmean(f1(A[..., 0], A[..., 1], A[..., 2]), 0)
    fb = np.nanmean(f1(Bc[..., 0], Bc[..., 1], Bc[..., 2]), 0)
    z = fa - fb
    pw = wilcoxon(fa, fb, method="exact").pvalue if np.any(z != 0) else 1.0
    return dict(dataset="DREAMS", n=len(recs), A=float(pooled(A)), B=float(pooled(Bc)), diff=float(d0),
                lo=float(np.percentile(db, 2.5)), hi=float(np.percentile(db, 97.5)),
                n_better=int((z > 0).sum()), n_worse=int((z < 0).sum()), p_wilcoxon=float(pw)), z


def nap_pair(nap, subjects, a_names, b_names, W):
    def integ(names, w):
        return np.mean([summarise(*nap[n], w)[1] for n in names], 0)

    def per_person(names):
        v = []
        for n in names:
            C, _ = nap[n]
            F = f1(C[:, :len(TAUS), 0], C[:, :len(TAUS), 1], C[:, :len(TAUS), 2])
            v.append(integrated(np.nan_to_num(F)))
        return np.mean(v, 0)

    one = np.ones((1, len(subjects)))
    a, b = integ(a_names, one)[0], integ(b_names, one)[0]
    db = integ(a_names, W) - integ(b_names, W)
    fa, fb = per_person(a_names), per_person(b_names)
    z = fa - fb
    pw = wilcoxon(fa, fb, method="exact").pvalue if np.any(z != 0) else 1.0
    return dict(dataset="NAP", n=len(subjects), A=float(a), B=float(b), diff=float(a - b),
                lo=float(np.percentile(db, 2.5)), hi=float(np.percentile(db, 97.5)),
                n_better=int((z > 0).sum()), n_worse=int((z < 0).sum()), p_wilcoxon=float(pw))


# ------------------------------------------------------------------ E. H4

def h4(seeds):
    lab = paths.CACHE / "labels" / "dreams_spindles"
    man = pd.read_csv(paths.CACHE / "manifest.csv")
    man = man[man.dataset == "dreams_spindles"]
    rows = []
    for r in man.itertuples():
        win = json.loads(r.windows)
        if "expert2" not in win:
            continue
        t1 = min(win["expert1"][1], win["expert2"][1])
        h = pd.read_csv(lab / f"{r.rec}__hypno.csv")
        ep = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
        st = h.stage.to_numpy()
        n2 = lambda t: st[np.clip((np.asarray(t) // ep).astype(int), 0, len(st) - 1)] == "N2"  # noqa: E731

        def ev(e):
            x = pd.read_csv(lab / f"{r.rec}__{e}__spindle.csv")
            a = np.c_[x.onset, x.onset + x.duration.fillna(0)].astype(float)
            a = a[(a[:, 0] >= 0) & (a[:, 0] < t1)]
            return a[n2((a[:, 0] + a[:, 1]) / 2)] if len(a) else a

        e1, e2 = ev("expert1"), ev("expert2")
        _, _, _, pairs = em.match(e1, e2, rule="onset", tol=0.5)
        m1, m2 = {i for i, _ in pairs}, {j for _, j in pairs}
        events = [(min(e1[i, 0], e2[j, 0]), max(e1[i, 1], e2[j, 1]), 0) for i, j in pairs]
        events += [(e1[i, 0], e1[i, 1], 1) for i in range(len(e1)) if i not in m1]
        events += [(e2[j, 0], e2[j, 1], 1) for j in range(len(e2)) if j not in m2]
        P = np.stack([np.load(ST / "work" / "gold_p" / s / "dreams_spindles" / f"{r.rec}.npz")["p"].astype(float)
                      for s in seeds])
        P = np.clip(P, 1e-6, 1 - 1e-6)
        pm = P.mean(0)
        H = -(pm * np.log(pm) + (1 - pm) * np.log(1 - pm))
        KL = (P * np.log(P / pm) + (1 - P) * np.log((1 - P) / (1 - pm))).mean(0)
        for a, b, one in events:
            i0, i1 = int(a * 100), max(int(np.ceil(b * 100)), int(a * 100) + 1)
            rows.append(dict(rec=r.rec, onset=a, offset=b, one_expert=one, p_mean=pm[i0:i1].mean(),
                             uncertainty=(H[i0:i1] + KL[i0:i1]).mean()))
    return pd.DataFrame(rows)


def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(gt)


# ------------------------------------------------------------------ main

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # A
    runs = section_a()
    runs.round(4).to_csv(OUT / "runs.csv", index=False)
    cols = ["val_f1_mv", "inj_f1", "inj_precision", "n2_per_min", "dur_median", "wake_over_n2", "best_pass",
            "passes_run", "hours"]
    arms = runs.groupby("arm")[cols].agg(["mean", "std"])
    arms.columns = [f"{a}_{b}" for a, b in arms.columns]

    # B
    s, p = dreams_tables()
    dre = []
    for label, ref, rule, tol in DREAMS_RULES:
        q = pick(s, ref, rule, tol).set_index("detector")
        for det in q.index:
            dre.append(dict(rule=label, detector=det, arm=arm_of(det), P=q.loc[det, "P_pooled"],
                            R=q.loc[det, "R_pooled"], F1=q.loc[det, "F1_pooled"]))
    dre = pd.DataFrame(dre)
    dre.round(4).to_csv(OUT / "dreams.csv", index=False)

    # C
    stores = {d.name: d for d in sorted((ST / "detections" / "nap").iterdir())}
    stores.update({t: RES / "detections_clean" / "nap" / t for t in TEACHERS})
    stores.update({o: RES / "detections" / "nap" / o for o in OPEN})
    nap, subjects = nap_scores(stores)
    W = np.random.default_rng(0).multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)),
                                             size=B).astype(float)
    nrows, pp = [], []
    for name, (C, aff) in nap.items():
        F, integ, P_, R_, AF = summarise(C, aff)
        _, binteg, *_ = summarise(C, aff, W)
        lo, hi = np.percentile(binteg, [2.5, 97.5])
        nrows.append(dict(detector=name, arm=arm_of(name), F1_integrated=integ[0], lo=lo, hi=hi,
                          F1_centre_0_5=F[0, KEYS.index("tau0.5")], aff_F1=AF[0],
                          P_0_5=None, n_det=None, **{f"F1_{k}": F[0, j] for j, k in enumerate(KEYS)}))
        for i, sj in enumerate(subjects):
            Fi = f1(C[i, :len(TAUS), 0], C[i, :len(TAUS), 1], C[i, :len(TAUS), 2])
            pp.append(dict(detector=name, subject=sj, F1_integrated=float(integrated(np.nan_to_num(Fi)))))
        tot = C.sum(0)
        k = KEYS.index("tau0.5")
        nrows[-1]["P_0_5"] = tot[k, 0] / max(tot[k, 0] + tot[k, 1], 1)
        nrows[-1]["n_det"] = int(tot[k, 0] + tot[k, 1])
    napdf = pd.DataFrame(nrows)
    napdf.round(4).to_csv(OUT / "nap.csv", index=False)
    pd.DataFrame(pp).round(4).to_csv(OUT / "nap_per_person.csv", index=False)

    # arm table with gold scores
    prim = dre[dre.rule == DREAMS_RULES[0][0]].set_index("detector")
    iou = dre[dre.rule == "E1 IoU 0.2"].set_index("detector")
    e2 = dre[dre.rule == "E2 onset 0.5"].set_index("detector")
    nn = napdf.set_index("detector")
    rows = []
    for arm, (label, group) in ARMS.items():
        seeds = sorted(d for d in prim.index if arm_of(d) == arm)
        v = prim.loc[seeds, "F1"]
        n = nn.loc[seeds, "F1_integrated"]
        rows.append(dict(arm=arm, label=label, group=group, n_seeds=len(seeds),
                         dreams_F1=v.mean(), dreams_min=v.min(), dreams_max=v.max(),
                         dreams_P=prim.loc[seeds, "P"].mean(), dreams_R=prim.loc[seeds, "R"].mean(),
                         dreams_iou02=iou.loc[seeds, "F1"].mean(), dreams_E2=e2.loc[seeds, "F1"].mean(),
                         nap_int=n.mean(), nap_min=n.min(), nap_max=n.max(),
                         nap_c05=nn.loc[seeds, "F1_centre_0_5"].mean(), nap_P05=nn.loc[seeds, "P_0_5"].mean(),
                         **{c: arms.loc[arm, c] for c in arms.columns}))
    for d, lab in {**TEACHERS, **CONSENSUS, **OPEN}.items():
        if d not in prim.index:
            continue
        grp = "teacher" if d in TEACHERS else "consensus" if d in CONSENSUS else "open baseline"
        rows.append(dict(arm=d, label=lab, group=grp, n_seeds=0, dreams_F1=prim.loc[d, "F1"],
                         dreams_P=prim.loc[d, "P"], dreams_R=prim.loc[d, "R"],
                         dreams_iou02=iou.loc[d, "F1"] if d in iou.index else np.nan,
                         dreams_E2=e2.loc[d, "F1"] if d in e2.index else np.nan,
                         nap_int=nn.loc[d, "F1_integrated"] if d in nn.index else np.nan,
                         nap_c05=nn.loc[d, "F1_centre_0_5"] if d in nn.index else np.nan,
                         nap_P05=nn.loc[d, "P_0_5"] if d in nn.index else np.nan))
    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT / "arms.csv", index=False)

    # D
    seeds_of = lambda arm: sorted(d for d in prim.index if arm_of(d) == arm)  # noqa: E731
    H = seeds_of(HEAD)
    best_nap_teacher = max(TEACHERS, key=lambda t: nn.loc[t, "F1_integrated"])
    hyp = []

    def add(hid, a_names, b_names, desc, kind="hypothesis", dreams=True, nap_=True):
        out = []
        if dreams:
            r, z = dreams_pair(p, a_names, b_names)
            out.append(r)
            if hid == "H5":
                pl, pr, prr = bayes_signed_rank(z)
                r.update(bayes_left=pl, bayes_rope=pr, bayes_right=prr)
        if nap_:
            out.append(nap_pair(nap, subjects, a_names, b_names, W))
        for r in out:
            r.update(id=hid, kind=kind, comparison=desc)
            hyp.append(r)

    add("H1", H, ["a7"], "headline vs A7 (DREAMS)", nap_=False)
    add("H1", H, [best_nap_teacher], f"headline vs best teacher on NAP ({TEACHERS[best_nap_teacher]})", dreams=False)
    add("H2", H, ["consensus_mv"], "headline vs teachers' majority vote")
    add("H3", H, seeds_of("student_soft_gpu"), "headline vs plain student on soft targets")
    add("H5", H, ["sumov2"], "headline vs SUMOv2 (ROPE +-0.02)")
    add("H3b", seeds_of("student_rules4_gpu"), seeds_of("student_soft_gpu"),
        "free noise model vs plain student on soft targets", kind="secondary")
    add("S", H, seeds_of("student_rules4_gpu"), "anchored vs free noise model", kind="secondary")
    for abl in ["student_rules4_lam0_gpu", "student_rules4_global_gpu", "student_rules4_noB_gpu",
                "student_rules4_nohead2_gpu"]:
        add("S", seeds_of(abl), seeds_of("student_rules4_gpu"), f"{ARMS[abl][0]} vs free noise model",
            kind="secondary")
    for st_, tgt in TARGET_OF.items():
        add("S", seeds_of(st_), [tgt], f"{ARMS[st_][0]} student vs its own target used as detector",
            kind="secondary")
    for st_, t in TEACHER_OF.items():
        add("S", seeds_of(st_), [t], f"student of {TEACHERS[t]} vs {TEACHERS[t]} itself", kind="secondary")
    hypdf = pd.DataFrame(hyp)
    hypdf.round(4).to_csv(OUT / "hypotheses.csv", index=False)

    # E
    ev = h4(H)
    ev.round(4).to_csv(OUT / "h4.csv", index=False)
    y = ev.one_expert.to_numpy()
    au_u = auroc(ev.uncertainty.to_numpy(), y)
    au_c = auroc(-ev.p_mean.to_numpy(), y)
    recs = sorted(ev.rec.unique())
    bs = []
    rng = np.random.default_rng(0)
    for _ in range(B):
        pick_ = rng.choice(recs, len(recs))
        q = pd.concat([ev[ev.rec == r_] for r_ in pick_])
        bs.append(auroc(q.uncertainty.to_numpy(), q.one_expert.to_numpy()))
    h4row = dict(n_events=len(ev), n_one_expert=int(y.sum()), n_both=int((y == 0).sum()),
                 auroc_uncertainty=au_u, lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)),
                 auroc_low_confidence=au_c)
    pd.DataFrame([h4row]).round(4).to_csv(OUT / "h4_summary.csv", index=False)

    write_md(table, runs, hypdf, h4row, best_nap_teacher)
    print(f"written {OUT / 'report.md'}")


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_md(t, runs, hyp, h4row, bnt):
    L = ["# GPU-track student grid: compiled results", "",
         "Generated by `experiments/student/report.py` from stored outputs. Frozen setting for every run: "
         "D0, τ 0.5 (amendment 10). DREAMS primary: expert 1, N2, onset ±0.5 s, first 1000 s, pooled F1. "
         "NAP primary: `nap_points.v1` integrated F1 (centre tolerance 0.25–1.5 s, N2, C3, 30 naps, 17 people). "
         "Student arms are means over 5 seeds (range in brackets).", "",
         "## Gold sets", "",
         "| group | detector | DREAMS F1 [seeds] | P | R | IoU≥0.2 | vs E2 | NAP int. F1 [seeds] | NAP F1 ±0.5 | NAP P ±0.5 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["teacher-aware", "ablation", "plain student", "single teacher", "consensus", "teacher", "open baseline"]
    t = t.assign(o=t.group.map(order.index)).sort_values(["o", "dreams_F1"], ascending=[True, False])
    for r in t.itertuples():
        rng_d = f" [{fmt(r.dreams_min)}–{fmt(r.dreams_max)}]" if r.n_seeds else ""
        rng_n = f" [{fmt(r.nap_min)}–{fmt(r.nap_max)}]" if r.n_seeds else ""
        L.append(f"| {r.group} | {r.label} | {fmt(r.dreams_F1)}{rng_d} | {fmt(r.dreams_P)} | {fmt(r.dreams_R)} | "
                 f"{fmt(r.dreams_iou02)} | {fmt(r.dreams_E2)} | {fmt(r.nap_int)}{rng_n} | {fmt(r.nap_c05)} | "
                 f"{fmt(r.nap_P05)} |")
    L += ["", "## Validation nights (no expert labels)", "",
          "| arm | agreement with vote (F1) | injected F1 | injected precision | N2 events/min | median dur. s | "
          "Wake/N2 | best pass | passes | GPU h per seed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in t[t.n_seeds > 0].itertuples():
        L.append(f"| {r.label} | {fmt(r.val_f1_mv_mean)} ± {fmt(r.val_f1_mv_std)} | {fmt(r.inj_f1_mean)} | "
                 f"{fmt(r.inj_precision_mean)} | {fmt(r.n2_per_min_mean, 2)} | {fmt(r.dur_median_mean, 2)} | "
                 f"{fmt(r.wake_over_n2_mean, 2)} | {fmt(r.best_pass_mean, 1)} | {fmt(r.passes_run_mean, 1)} | "
                 f"{fmt(r.hours_mean, 1)} |")
    L += ["", "## Pre-registered hypotheses and secondary contrasts", "",
          "Difference = A − B (seed-averaged). DREAMS: pooled F1, recording bootstrap; NAP: integrated F1, paired "
          "person bootstrap. Wilcoxon on per-recording / per-person F1 (exact). `better/worse` counts recordings or people.",
          "", "| id | comparison | set | A | B | diff [95% CI] | better/worse | p (Wilcoxon) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in hyp.itertuples():
        L.append(f"| {r.id} | {r.comparison} | {r.dataset} | {fmt(r.A)} | {fmt(r.B)} | {r.diff:+.3f} "
                 f"[{r.lo:+.3f}, {r.hi:+.3f}] | {r.n_better}/{r.n_worse} | {r.p_wilcoxon:.4f} |")
    h5 = hyp[(hyp.id == "H5") & (hyp.dataset == "DREAMS")]
    if len(h5) and "bayes_rope" in h5:
        r = h5.iloc[0]
        L += ["", f"H5 Bayesian signed-rank on DREAMS (ROPE ±{ROPE}): P(SUMOv2 better) {r.bayes_left:.3f}, "
                  f"P(practically equal) {r.bayes_rope:.3f}, P(student better) {r.bayes_right:.3f}."]
    L += ["", "## H4: ensemble uncertainty against expert disagreement (DREAMS excerpts 1–6)", "",
          f"{h4row['n_events']} expert events in N2 (both experts {h4row['n_both']}, one expert only "
          f"{h4row['n_one_expert']}). AUROC of ensemble uncertainty for 'one expert only': "
          f"**{h4row['auroc_uncertainty']:.3f}** [{h4row['lo']:.3f}, {h4row['hi']:.3f}] (excerpt bootstrap); "
          f"pre-registered threshold 0.7. AUROC of low mean probability: {h4row['auroc_low_confidence']:.3f}.",
          "", f"Best rule teacher on NAP (for H1): {TEACHERS[bnt]}."]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
