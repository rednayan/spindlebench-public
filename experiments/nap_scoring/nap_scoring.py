"""
NAP spindle scoring with documented methods only (reports/Scoring point annotated events.md;
decision 2026-10-28, see docs/thesis/04_evaluation_protocol.md).

    uv run python experiments/nap_scoring/nap_scoring.py [--draws 10] [--workers 12]

NAP marks are single points (no onset, no duration). Every detector is scored three ways:

1. Tolerance matching + tolerance sweep (primary).
   One-to-one (Hungarian) matching of detection centre to mark within +-tau, as in ECG beat matching
   (ANSI/AAMI EC57, bxb), SpikeForest and TCPDBench. tau on a grid fixed in advance:
   0.25, 0.50, ..., 1.50 s. Pooled F1 at each tau, and the integrated F1 = area under F1(tau) over
   [0.25, 1.5] s divided by its width (trapezoid), following the "F1 integrated over thresholds" of
   SUMO (Kaulen et al. 2022) and the area-under-curve summaries of Kim et al. (AAAI 2022).
   The pre-specified +-0.5 s result is one point of the curve.
2. Affiliation precision / recall (Huet et al., KDD 2022), computed with the authors' library
   (affiliation-metrics-py, commit 8d84498, MIT; installed as a locked dependency). The library rejects zero-length
   events; following the paper's Appendix B (a point at t is the limit of [t, t+eps)), each mark is
   passed as [t, t + 0.01 s] (one 100 Hz sample). Per nap: Trange = (0, end of the nap);
   affiliation P and R are averaged over naps (P over naps where it is defined); affiliation F1 is
   their harmonic mean.
3. Containment check (not primary): the mark lies inside the detection widened by 0.25 s on each
   side, one-to-one ("pointing game"-style hit, Zhang et al. ECCV 2016).

Scope N2 as in the main protocol: marks in expert N2 epochs, detections whose midpoint is in N2.
30 included naps, channel C3.

Random baseline (Kim et al. 2022; Huet et al. 2022): for every detector and nap, the same number of
detections with the same durations placed uniformly at random (midpoint in N2, non-overlapping);
`--draws` draws, scored with 1-3.

Uncertainty: person-level bootstrap (both naps of a person together, 2000 resamples, seed 0).
Detectors: the NAP benchmark store (stage-free spindle detectors) and the SUMOv2 injection arms.
-> results/nap_scoring/{scores.csv, scores_random.csv, arm_contrasts.csv, rank_agreement.csv, f1_vs_tolerance.png}
"""

import argparse
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from spindlebench import paths, registry
from spindlebench.data.nap import in_n2, load_det, nap_info, naps  # noqa: F401  (used by mass_raters, nap_rater_rule)
from spindlebench.protocols.nap_points import (  # noqa: F401
    KEYS,
    MARGIN,
    TAUS,
    f1,
    integrated,
    random_dets,
    score_nap,
    summarise,
    tensors,
)
from spindlebench import provenance  # noqa: E402
from spindlebench.protocols import nap_points  # noqa: E402

provenance.track(__file__, protocols=[nap_points.VERSION])

ROOT = paths.ROOT
OUT = paths.RESULTS / "nap_scoring"
EXCLUDE = registry.NAP_EXCLUDE
STORES = {"benchmark": paths.RESULTS / "detections" / "nap",
          "injection": paths.RESULTS / "injection" / "detections" / "nap"}
B = 2000


def job(args):
    store, det, info, draws = args
    on, off = load_det(STORES[store] / det / f"{info['rec']}.csv", info)
    real = score_nap(on, off, info["marks"], info["T"])
    rng = np.random.default_rng(zlib.crc32(f"{det}|{info['rec']}".encode()))
    rand = [score_nap(*random_dets(on, off, info, rng), info["marks"], info["T"]) for _ in range(draws)]
    return store, det, info["rec"], info["subject"], real, rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    infos = [nap_info(r) for r in naps().itertuples()]
    subjects = sorted({i["subject"] for i in infos})
    dets = [("benchmark", d.name) for d in sorted(STORES["benchmark"].iterdir()) if d.name not in EXCLUDE]
    dets += [("injection", d.name) for d in sorted(STORES["injection"].iterdir()) if d.name != "sumov2_zeroshot"]
    jobs = [(s, d, i, a.draws) for s, d in dets for i in infos]
    print(len(dets), "detectors,", len(jobs), "jobs", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(job, jobs, chunksize=4))

    rng = np.random.default_rng(0)
    W = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=B).astype(float)
    by = {}
    for store, det, rec, subj, real, rand in res:
        by.setdefault((store, det), []).append((subj, real, rand))
    rows, rrows, boot = [], [], {}
    for (store, det), lst in by.items():
        C, aff = tensors([(s, r) for s, r, _ in lst], subjects)
        F, integ, P, R, AF = summarise(C, aff)
        bF, binteg, bP, bR, bAF = summarise(C, aff, W)
        boot[det] = dict(integ=binteg, AF=bAF, F05=bF[:, 1], contain=bF[:, -1])
        row = dict(store=store, detector=det)
        row.update({f"F1_{k}": F[0, j] for j, k in enumerate(KEYS)})
        lo, hi = np.percentile(binteg, [2.5, 97.5])
        row.update(F1_integrated=integ[0], F1_integrated_lo=lo, F1_integrated_hi=hi,
                   aff_P=P[0], aff_R=R[0], aff_F1=AF[0])
        rows.append(row)
        # random baseline: average over draws of each statistic
        draws = len(lst[0][2])
        acc = []
        for k in range(draws):
            Ck, affk = tensors([(s, rd[k]) for s, _, rd in lst], subjects)
            Fk, ik, Pk, Rk, AFk = summarise(Ck, affk)
            acc.append(np.r_[Fk[0], ik[0], Pk[0], Rk[0], AFk[0]])
        acc = np.array(acc)
        rr = dict(store=store, detector=det, draws=draws)
        rr.update({f"F1_{k}": acc[:, j].mean() for j, k in enumerate(KEYS)})
        n = len(KEYS)
        rr.update(F1_integrated=acc[:, n].mean(), F1_integrated_sd=acc[:, n].std(), aff_P=acc[:, n + 1].mean(),
                  aff_R=acc[:, n + 2].mean(), aff_F1=acc[:, n + 3].mean())
        rrows.append(rr)
    sc = pd.DataFrame(rows)
    rs = pd.DataFrame(rrows)
    sc.round(4).to_csv(OUT / "scores.csv", index=False)
    rs.round(4).to_csv(OUT / "scores_random.csv", index=False)

    # SUMOv2 arm contrasts (3-seed means), every scoring
    arms = {"A0 zero-shot": ["sumov2"], "B self-training": ["sumov2_self", "sumov2_self_s1", "sumov2_self_s2"],
            "C injection": ["sumov2_inj", "sumov2_inj_s1", "sumov2_inj_s2"],
            "D real labels": ["sumov2_real", "sumov2_real_s1", "sumov2_real_s2"]}
    contrasts = [("C injection", "A0 zero-shot"), ("C injection", "B self-training"),
                 ("C injection", "D real labels"), ("D real labels", "A0 zero-shot")]
    con = []
    pt = sc.set_index("detector")
    for metric, col in [("F05", "F1_tau0.5"), ("integ", "F1_integrated"), ("AF", "aff_F1"), ("contain", "F1_contain")]:
        for x, y in contrasts:
            bx = np.mean([boot[d][metric] for d in arms[x]], axis=0)
            by_ = np.mean([boot[d][metric] for d in arms[y]], axis=0)
            dpt = pt.loc[arms[x], col].mean() - pt.loc[arms[y], col].mean()
            lo, hi = np.nanpercentile(bx - by_, [2.5, 97.5])
            con.append(dict(metric=col, contrast=f"{x} - {y}", diff=dpt, lo=lo, hi=hi, excludes_0=bool(lo > 0 or hi < 0)))
    con = pd.DataFrame(con)
    con.round(4).to_csv(OUT / "arm_contrasts.csv", index=False)

    # ranking agreement among benchmark detectors
    bm = sc[sc.store == "benchmark"].set_index("detector")
    ra = []
    for col in ["F1_tau1.0", "F1_integrated", "aff_F1", "F1_contain"]:
        t, p = kendalltau(bm["F1_tau0.5"], bm[col])
        ra.append(dict(vs="F1_tau0.5", metric=col, n=len(bm), tau=t, p=p))
    t, p = kendalltau(bm["F1_integrated"], bm["aff_F1"])
    ra.append(dict(vs="F1_integrated", metric="aff_F1", n=len(bm), tau=t, p=p))
    pd.DataFrame(ra).round(4).to_csv(OUT / "rank_agreement.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["detector", "F1_tau0.25", "F1_tau0.5", "F1_tau1.0", "F1_tau1.5", "F1_integrated", "F1_integrated_lo",
            "F1_integrated_hi", "aff_P", "aff_R", "aff_F1", "F1_contain"]
    print(sc[sc.store == "benchmark"].sort_values("F1_integrated", ascending=False)[cols].round(3).to_string(index=False))
    print(sc[sc.store == "injection"].sort_values("detector")[cols].round(3).to_string(index=False))
    print("random baseline:")
    print(rs[rs.store == "benchmark"].sort_values("F1_integrated", ascending=False)[
        ["detector", "F1_tau0.5", "F1_integrated", "F1_integrated_sd", "aff_P", "aff_R", "aff_F1", "F1_contain"]].round(3).to_string(index=False))
    print(con.round(3).to_string(index=False))
    print(pd.DataFrame(ra).round(3).to_string(index=False))
    plot(sc, rs)


def plot(sc, rs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = {"sumov2": ("SUMOv2", "#1F4E79"), "sumo": ("SUMO", "#2E75B6"), "a7": ("A7", "#E07B39"),
            "moelle2011_faithful": ("Mölle", "#6A9F3F"), "yasa_dreams_tuned": ("YASA tuned", "#8E5EA2"),
            "openspindlenet": ("OpenSpindleNet", "#C0504D"), "yasa_default": ("YASA default", "#9E9E9E")}
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    cols = [f"F1_tau{t}" for t in TAUS]
    for d, (name, col) in show.items():
        r = sc[(sc.store == "benchmark") & (sc.detector == d)]
        q = rs[(rs.store == "benchmark") & (rs.detector == d)]
        if len(r):
            ax.plot(TAUS, r[cols].to_numpy()[0], "-o", ms=4, lw=2, color=col, label=name)
            ax.plot(TAUS, q[cols].to_numpy()[0], ":", lw=1.2, color=col)
    ax.axvline(0.5, color="#999", lw=0.8, ls="--")
    ax.text(0.52, 0.02, "pre-specified ±0.5 s", fontsize=8, color="#666")
    ax.set_xlabel("Matching tolerance τ (s), detection centre vs mark")
    ax.set_ylabel("Pooled N2 F1 (NAP)")
    ax.set_title("F1 vs tolerance (solid) and random baseline (dotted)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(OUT / "f1_vs_tolerance.png", dpi=130)


if __name__ == "__main__":
    main()
