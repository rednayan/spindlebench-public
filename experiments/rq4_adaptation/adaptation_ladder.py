"""
RQ4: the adaptation ladder on NAP, cheapest rung first, with paired uncertainty.

    uv run python experiments/rq4_adaptation/adaptation_ladder.py

Inputs
  results/rq4/benchmark/per_recording_partial.csv   SUMO rungs (experiments/rq1_benchmark/evaluate.py --store results/rq4/detections)
  results/benchmark/per_recording.csv       benchmark reference rows (Mölle 2011, SUMOv2 zero-shot)
  results/adaptation/nap_threshold_adaptation_per_recording.csv   YASA threshold rungs
All on the included NAP naps, C3, centre ±0.5 s vs the expert, N2 and whole nap.

Outputs (results/rq4/)
  ladder_summary.csv   pooled F1 per rung and scope, with a subject-bootstrap 95% CI
  ladder_paired.csv    paired subject-level bootstrap of F1 differences between rungs (95% CI;
                       same naps on both sides, both naps of a subject resampled together).
                       Three contrasts are primary (zero-shot -> fine-tuned, fine-tuned ->
                       scratch, Mölle -> scratch; 3 seeds pooled) and also get a Bonferroni-
                       adjusted 98.3% CI; every other row is exploratory
  ladder_seed_pairs.csv  every scratch seed vs every fine-tuning seed (lr 1e-3)
  ladder.png           bars: seed-pooled rungs where 3 seeds exist (dots = single seeds),
                       whiskers: subject-bootstrap 95% CI
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spindlebench import paths, plotting
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

OUT = paths.RESULTS / "rq4"
RNG = np.random.default_rng(0)
SUMO_RUNGS = ["sumo_zeroshot", "sumo_recal", "sumo_persubj", "sumo_ft_n1", "sumo_ft_n3", "sumo_ft_n6",
              "sumo_ft_all", "sumo_scratch"]
# extra seeds (s1, s2) of ft_all and scratch, drawn as dots on the seed-0 bar; lr sensitivity rung
SEEDS = {"sumo_ft_all": ["sumo_ft_all_s1", "sumo_ft_all_s2"], "sumo_scratch": ["sumo_scratch_s1", "sumo_scratch_s2"],
         # lr control: fine-tuning at scratch's lr 5e-3, same seeds, so ft vs scratch differs only in the init
         "sumo_ft_all_lr5e-3": ["sumo_ft_all_s1_lr5e-3", "sumo_ft_all_s2_lr5e-3"]}
EXTRA = ["sumo_ft_all_lr5e-3"]
REFERENCE = ["wonambi_moelle2011", "sumov2"]


def load():
    rq4 = pd.read_csv(OUT / "benchmark" / "per_recording_partial.csv")
    bench = pd.read_csv(paths.RESULTS / "benchmark/per_recording.csv")
    keep = lambda d: d[(d.dataset == "nap") & (d.event == "spindle") & (d.reference == "expert")
                       & (d.rule == "center") & (d.tol == 0.5) & (~d.excluded)]
    rows = pd.concat([keep(rq4[~rq4.detector.str.endswith("_t05")]), keep(bench[bench.detector.isin(REFERENCE)])])
    rows = rows[["rec", "detector", "scope", "tp", "fp", "fn"]].rename(columns={"detector": "rung"})
    y = pd.read_csv(paths.RESULTS / "adaptation/nap_threshold_adaptation_per_recording.csv")
    y = y.assign(rec=[f"suj{s}_d{d}" for s, d in zip(y.subject, y.day)], rung="yasa_" + y.strategy)
    # seed-pooled rungs: tp/fp/fn summed over the 3 seeds of a nap, so one lucky seed can't carry a claim
    pooled = [rows[rows.rung.isin([b] + extra)].groupby(["rec", "scope"])[["tp", "fp", "fn"]].sum()
              .reset_index().assign(rung=b + "_3seed") for b, extra in SEEDS.items()
              if set(extra) <= set(rows.rung)]
    # Mölle 2011 (paper recipe) with its threshold factor recalibrated (experiments/rq4_adaptation/adaptation_moelle.py)
    mp = paths.RESULTS / "adaptation/moelle_threshold_adaptation_per_recording.csv"
    mo = pd.read_csv(mp) if mp.exists() else y.iloc[:0]
    mo = mo.assign(rec=[f"suj{s}_d{d}" for s, d in zip(mo.subject, mo.day)], rung="moelle_" + mo.strategy)
    return pd.concat([rows, *pooled, y[["rec", "rung", "scope", "tp", "fp", "fn"]],
                      mo[["rec", "rung", "scope", "tp", "fp", "fn"]]])


def f1(t):
    return 2 * t[0] / (2 * t[0] + t[1] + t[2]) if t.sum() else np.nan


PRIMARY = [("sumo_zeroshot", "sumo_ft_all_3seed"), ("sumo_ft_all_3seed", "sumo_scratch_3seed"),
           ("wonambi_moelle2011", "sumo_scratch_3seed")]


def subject(rec):
    return rec.split("_")[0]


def by_subject(df, rung, recs):
    """tp/fp/fn summed over the naps of each subject (naps restricted to recs)"""
    d = df[(df.rung == rung) & df.rec.isin(recs)]
    return d.assign(subj=d.rec.map(subject)).groupby("subj")[["tp", "fp", "fn"]].sum()


def f1_rows(x):
    with np.errstate(invalid="ignore"):
        return 2 * x[:, 0] / (2 * x[:, 0] + x[:, 1] + x[:, 2])


def paired(df, a, b, n=2000, alpha=0.05):
    """subject-level bootstrap over naps that both rungs scored: F1(b) - F1(a)"""
    recs = set(df[df.rung == a].rec) & set(df[df.rung == b].rec)
    A, B = by_subject(df, a, recs), by_subject(df, b, recs)
    A, B = A.to_numpy(float), B.loc[A.index].to_numpy(float)
    d0 = f1(B.sum(0)) - f1(A.sum(0))
    idx = RNG.integers(0, len(A), (n, len(A)))
    ds = f1_rows(B[idx].sum(1)) - f1_rows(A[idx].sum(1))
    return len(recs), d0, np.nanquantile(ds, alpha / 2), np.nanquantile(ds, 1 - alpha / 2)


def rung_ci(df, rung, recs, n=2000):
    x = by_subject(df, rung, recs).to_numpy(float)
    idx = RNG.integers(0, len(x), (n, len(x)))
    v = f1_rows(x[idx].sum(1))
    return np.nanquantile(v, 0.025), np.nanquantile(v, 0.975)


def main():
    df = load()
    summ = (df.groupby(["scope", "rung"])[["tp", "fp", "fn"]].sum()
            .assign(F1=lambda t: 2 * t.tp / (2 * t.tp + t.fp + t.fn),
                    n_rec=df.groupby(["scope", "rung"]).rec.nunique()).reset_index())
    # same naps for every rung (YASA rungs cover only the 26 naps whose subject has both naps included)
    common = set.intersection(*[set(g.rec) for _, g in df.groupby("rung")])
    dc = df[df.rec.isin(common)]
    summ_c = (dc.groupby(["scope", "rung"])[["tp", "fp", "fn"]].sum()
              .assign(F1_common=lambda t: 2 * t.tp / (2 * t.tp + t.fp + t.fn)).reset_index())
    summ = summ.merge(summ_c[["scope", "rung", "F1_common"]], on=["scope", "rung"], how="left")
    ci = [dict(scope=sc, rung=r, **dict(zip(["common_lo", "common_hi"], rung_ci(df[df.scope == sc], r, common))))
          for sc, r in summ[["scope", "rung"]].itertuples(index=False)]
    summ = summ.merge(pd.DataFrame(ci), on=["scope", "rung"], how="left")
    summ.to_csv(OUT / "ladder_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"F1 on each rung's own naps, and F1_common on the {len(common)} naps every rung scored")
    print(summ.pivot(index="rung", columns="scope", values=["F1", "F1_common"]).round(3)
          .sort_values(("F1_common", "N2")).to_string())

    present = [r for r in SUMO_RUNGS if r in set(df.rung)]
    pairs = list(zip(present[:-1], present[1:])) + [("sumo_zeroshot", r) for r in present[2:]] \
        + [("sumo_zeroshot", "sumo_ft_all_lr5e-3"), ("sumo_ft_all", "sumo_ft_all_lr5e-3"),
           ("sumo_ft_all_lr5e-3", "sumo_scratch"), ("sumo_ft_all", "sumo_scratch"),
           ("wonambi_moelle2011", "sumo_scratch"), ("yasa_oracle", "sumo_scratch"),
           ("wonambi_moelle2011", "sumo_ft_all"), ("yasa_oracle", "sumo_ft_all"),
           ("sumo_zeroshot", "sumo_ft_all_3seed"), ("sumo_zeroshot", "sumo_scratch_3seed"),
           ("sumo_ft_all_3seed", "sumo_scratch_3seed"), ("wonambi_moelle2011", "sumo_scratch_3seed"),
           ("yasa_oracle", "sumo_scratch_3seed"), ("yasa_population", "sumo_scratch_3seed"),
           ("moelle_default", "moelle_population"), ("moelle_population", "sumo_scratch_3seed"),
           # lr-matched: released MODA init vs random init at the same lr 5e-3 and seeds
           ("sumo_ft_all_lr5e-3_3seed", "sumo_scratch_3seed"), ("sumo_ft_all_3seed", "sumo_ft_all_lr5e-3_3seed")]
    rows = []
    for scope in ("N2", "all"):
        d = df[df.scope == scope]
        for a, b in dict.fromkeys(pairs):
            if a in set(d.rung) and b in set(d.rung):
                n, diff, lo, hi = paired(d, a, b)
                row = dict(scope=scope, a=a, b=b, n_rec=n, diff_b_minus_a=diff, lo=lo, hi=hi,
                           resolved=(lo > 0) or (hi < 0), primary=(a, b) in PRIMARY)
                if (a, b) in PRIMARY:
                    _, _, blo, bhi = paired(d, a, b, alpha=0.05 / len(PRIMARY))
                    row.update(lo_bonf=blo, hi_bonf=bhi, resolved_bonf=(blo > 0) or (bhi < 0))
                rows.append(row)
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / "ladder_paired.csv", index=False)
    print(pr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # seed level: every scratch seed against every fine-tuning seed (same lr 1e-3 recipe otherwise)
    sp = []
    for scope in ("N2", "all"):
        t = summ[summ.scope == scope].set_index("rung").F1
        for ft in ("sumo_ft_all", "sumo_ft_all_lr5e-3"):
            for a in [ft] + SEEDS[ft]:
                for b in ["sumo_scratch"] + SEEDS["sumo_scratch"]:
                    if a in t.index and b in t.index:
                        sp.append(dict(scope=scope, ft_lr="1e-3" if ft == "sumo_ft_all" else "5e-3", ft_seed=a,
                                       scratch_seed=b, F1_ft=t[a], F1_scratch=t[b], scratch_better=t[b] > t[a]))
    sp = pd.DataFrame(sp)
    sp.to_csv(OUT / "ladder_seed_pairs.csv", index=False)
    for (scope, lr), g in sp.groupby(["scope", "ft_lr"]):
        print(f"{scope}, fine-tuning lr {lr}: scratch seed beats fine-tuning seed in {g.scratch_better.sum()}/{len(g)} pairs")

    # seed-pooled bars where 3 seeds exist, so no single (lucky) seed stands for the rung
    bar = {r: r + "_3seed" for r in SEEDS if r + "_3seed" in set(summ.rung)}
    order = ["yasa_default", "yasa_transfer", "yasa_population", "yasa_per_subject", "yasa_oracle",
             "sumov2"] + [bar.get(r, r) for r in present[:-1]] + [bar.get(r, r) for r in EXTRA] \
        + [bar.get(r, r) for r in present[-1:]] \
        + ["moelle_default", "moelle_population", "moelle_oracle", "wonambi_moelle2011"]
    s = summ[summ.scope == "N2"].set_index("rung").reindex([o for o in order if o in set(summ.rung)])
    fig, ax = plt.subplots(figsize=(13, 4.4))
    cols = [plotting.C_YASA if r.startswith("yasa") else (plotting.C_E1 if r.startswith("sumo") and r not in REFERENCE else "#a7a6a0") for r in s.index]
    ax.bar(range(len(s)), s.F1_common, color=cols, edgecolor="white",
           yerr=[s.F1_common - s.common_lo, s.common_hi - s.F1_common], error_kw=dict(ecolor=plotting.INK2, lw=1, capsize=2))
    sc = summ[summ.scope == "N2"].set_index("rung").F1_common
    for i, (r, v) in enumerate(s.F1_common.items()):
        base = r.removesuffix("_3seed")
        seeds = [sc[x] for x in [base] + SEEDS.get(base, []) if r.endswith("_3seed") and x in sc.index]
        ax.scatter([i] * len(seeds), seeds, s=18, color=plotting.INK, zorder=3)
        ax.text(i, s.common_hi[r] + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=plotting.INK)
    names = [r.replace("wonambi_", "").replace("_3seed", "\n(3 seeds)").replace("_", "\n") for r in s.index]
    ax.set_xticks(range(len(s)), names, fontsize=7.5)
    ax.set_ylabel(f"pooled F1, N2\n(same {len(common)} naps, C3, centre ±0.5 s)", fontsize=9)
    ax.set_ylim(0, max(0.6, s.common_hi.max() + 0.08))
    ax.yaxis.grid(True, color=plotting.GRID, lw=0.8)
    ax.set_axisbelow(True)
    plotting.style(ax)
    ax.set_title("Adaptation ladder: YASA thresholds (blue), SUMO rungs (orange), references (grey); "
                 "dots = single seeds; whiskers = subject-bootstrap 95% CI", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ladder.png", dpi=170)
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()
