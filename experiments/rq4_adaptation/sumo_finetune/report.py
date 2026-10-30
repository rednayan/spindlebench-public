"""
Step 6: RQ4 tables from the evaluator output (main project venv).

    uv run python experiments/rq1_benchmark/evaluate.py --store results/rq4/detections --datasets nap,dreams_spindles --out results/rq4/benchmark
    uv run python experiments/rq4_adaptation/sumo_finetune/report.py

NAP: centre +-0.5 s vs expert, 30 included naps, pooled F1 (scope N2 and all), with a
subject-level paired bootstrap (both naps of a subject together; 2000 resamples, seed 0) of the
difference to sumo_zeroshot.
DREAMS: onset +-0.5 s vs expert 1 (main protocol) and centre +-0.5 s (secondary: the fine-tuned
models learnt peak-centred +-0.4 s intervals, so their onsets are biased late by construction).
Each fine-tuned id is the 5-fold ensemble; the mean and range of its single fold models
(<id>_fold<k>) separate NAP training from ensembling. dF1_vs_sumov2: excerpt-bootstrap 95% CI of
F1(id) - F1(SUMOv2, benchmark store) on the same 8 excerpts.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "results" / "rq4" / "benchmark"
HERE = Path(__file__).resolve().parent
ORDER = ["sumo_zeroshot", "sumo_recal", "sumo_persubj", "sumo_ft_n1", "sumo_ft_n3", "sumo_ft_n6", "sumo_ft_all",
         "sumo_ft_all_s1", "sumo_ft_all_s2", "sumo_ft_all_lr5e-3", "sumo_scratch", "sumo_scratch_s1", "sumo_scratch_s2"]
RNG = np.random.default_rng(0)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def boot_diff(g, a, b, n=2000, unit=lambda rec: rec.split("_")[0]):
    """paired bootstrap of F1(a) - F1(b) over units (NAP: subject = both naps; DREAMS: pass unit=str)"""
    g = g.assign(unit=g.rec.map(unit))
    piv = {c: g.pivot_table(index="unit", columns="detector", values=c, aggfunc="sum") for c in ("tp", "fp", "fn")}
    units = piv["tp"].index.to_numpy()
    d = []
    for _ in range(n):
        s = RNG.choice(units, len(units), replace=True)
        t = {c: piv[c].loc[s].sum() for c in piv}
        d.append(f1(t["tp"][a], t["fp"][a], t["fn"][a]) - f1(t["tp"][b], t["fp"][b], t["fn"][b]))
    return np.percentile(d, [2.5, 97.5])


def main():
    per = pd.read_csv(BENCH / "per_recording_partial.csv")
    per = per[(per.event == "spindle")]
    nap = per[(per.dataset == "nap") & (~per.excluded) & (per.reference == "expert") & (per.rule == "center")
              & (per.tol == 0.5)]
    ids = [i for i in ORDER if i in set(nap.detector)] + sorted(set(nap.detector) - set(ORDER))
    rows = []
    for did in ids:
        r = dict(id=did)
        for scope in ("N2", "all"):
            g = nap[(nap.detector == did) & (nap.scope == scope)]
            r[f"n_rec_{scope}"] = len(g)
            r[f"P_{scope}"] = g.tp.sum() / max(g.tp.sum() + g.fp.sum(), 1)
            r[f"R_{scope}"] = g.tp.sum() / max(g.tp.sum() + g.fn.sum(), 1)
            r[f"F1_{scope}"] = f1(g.tp.sum(), g.fp.sum(), g.fn.sum())
            if did != "sumo_zeroshot":
                lo, hi = boot_diff(nap[nap.scope == scope], did, "sumo_zeroshot")
                r[f"dF1_{scope}_CI"] = f"[{lo:+.3f}, {hi:+.3f}]"
            if scope == "N2":
                z = nap[(nap.detector == "sumo_zeroshot") & (nap.scope == "N2")].set_index("rec").f1
                x = g.set_index("rec").f1
                d = (x - z).dropna()
                r["better_naps_N2"] = f"{(d > 0).sum()}/{len(d)}"
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(BENCH / "rq4_nap_table.csv", index=False)
    pd.set_option("display.width", 250)
    print("NAP (centre +-0.5 s vs expert, 30 included naps, pooled):")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for base in ("sumo_ft_all", "sumo_scratch"):
        v = t[t.id.isin([base, base + "_s1", base + "_s2"])]
        if len(v) > 1:
            print(f"seed spread {base}: N2 {v.F1_N2.min():.3f}-{v.F1_N2.max():.3f} (mean {v.F1_N2.mean():.3f}, "
                  f"sd {v.F1_N2.std():.3f}); all {v.F1_all.min():.3f}-{v.F1_all.max():.3f} (mean {v.F1_all.mean():.3f})")

    dr = per[(per.dataset == "dreams_spindles") & (per.reference == "expert1") & (per.tol == 0.5)
             & per.rule.isin(["onset", "center"])]
    drows = []
    for (did, rule, scope), g in dr.groupby(["detector", "rule", "scope"]):
        drows.append(dict(id=did, rule=rule, scope=scope, n_det=int(g.tp.sum() + g.fp.sum()),
                          P=g.tp.sum() / max(g.tp.sum() + g.fp.sum(), 1), R=g.tp.sum() / (g.tp.sum() + g.fn.sum()),
                          F1=f1(g.tp.sum(), g.fp.sum(), g.fn.sum())))
    d = pd.DataFrame(drows)
    # single fold models: mean and range per ensemble id
    fold = d[d.id.str.contains(r"_fold\d$", regex=True)].assign(base=lambda x: x.id.str.replace(r"_fold\d$", "", regex=True))
    agg = fold.groupby(["base", "rule", "scope"]).F1.agg(["mean", "min", "max"]).rename(
        columns={"mean": "F1_single_mean", "min": "F1_single_min", "max": "F1_single_max"})
    d = d[~d.id.str.contains(r"_fold\d$", regex=True)].merge(agg, left_on=["id", "rule", "scope"], right_index=True, how="left")
    # ensemble vs SUMOv2 from the benchmark store, same excerpts
    bench = pd.read_csv(ROOT / "results" / "benchmark" / "per_recording.csv")
    bench = bench[(bench.dataset == "dreams_spindles") & (bench.detector == "sumov2") & (bench.reference == "expert1")
                  & (bench.tol == 0.5) & bench.rule.isin(["onset", "center"])]
    both = pd.concat([dr, bench[dr.columns]])
    ci = []
    for r in d.itertuples():
        g = both[(both.rule == r.rule) & (both.scope == r.scope) & both.detector.isin([r.id, "sumov2"])]
        if g.detector.nunique() == 2:
            lo, hi = boot_diff(g, r.id, "sumov2", unit=str)
            ci.append(f"[{lo:+.3f}, {hi:+.3f}]")
        else:
            ci.append("")
    d["dF1_vs_sumov2_CI"] = ci
    d = d.sort_values(["rule", "scope", "id"], ascending=[False, False, True])
    d.to_csv(BENCH / "rq4_dreams_table.csv", index=False)
    print("\nDREAMS (vs expert 1, window 0-1000 s, pooled over 8 excerpts):")
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # learning curve figure
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    xs = {"sumo_ft_n1": 1, "sumo_ft_n3": 3, "sumo_ft_n6": 6, "sumo_ft_all": 11.4}
    for scope, c in (("N2", "#2a78d6"), ("all", "#eb6834")):
        v = t.set_index("id")
        ks = [k for k in xs if k in v.index]
        ax.plot([xs[k] for k in ks], [v.loc[k, f"F1_{scope}"] for k in ks], "o-", color=c, label=f"fine-tuned ({scope})")
        ax.axhline(v.loc["sumo_zeroshot", f"F1_{scope}"], color=c, ls=":", lw=1)
        if "sumo_scratch" in v.index:
            ax.plot([11.4], [v.loc["sumo_scratch", f"F1_{scope}"]], "s", mfc="white", color=c, ms=7,
                    label=f"from scratch, all ({scope})")
    summ = pd.read_csv(ROOT / "results" / "benchmark" / "summary.csv")
    molle = summ.query("dataset == 'nap' and event == 'spindle' and detector == 'wonambi_moelle2011' and "
                       "reference == 'expert' and rule == 'center' and tol == 0.5 and scope == 'N2'").F1_pooled.iloc[0]
    ax.axhline(molle, color="#52514e", ls="--", lw=1)
    ax.text(1, molle + 0.004, "Mölle 2011, wonambi (N2)", fontsize=8, color="#52514e", va="bottom")
    ax.set(xlabel="training subjects (all = 11-12 per fold)", ylabel="pooled F1, centre ±0.5 s",
           title="SUMO on NAP C3: zero-shot (dotted) vs fine-tuned", xscale="log")
    ax.set_xticks([1, 3, 6, 11.4], ["1", "3", "6", "all"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e4e3df", lw=0.8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "curves" / "learning_curve.png", dpi=150)


if __name__ == "__main__":
    main()
