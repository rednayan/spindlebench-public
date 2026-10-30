"""
E2 (RQ2): does the detector ranking survive the evaluation choices?

    uv run python experiments/rq2_validity/validity_e2.py [--per results/benchmark/per_recording.csv] [--min-detectors 3]

Input: the long table written by experiments/rq1_benchmark/evaluate.py (every detector x recording x reference x
scope x rule x tolerance). For each dataset/event, for the primary protocol of RQ1 (N2
scope) and its whole-recording counterpart, and for two detector pools (every detector;
out-of-sample only, i.e. without detectors trained or tuned on that dataset's labels):
  1. pooled F1 per detector under every protocol variant
     (reference expert1/expert2/union/intersection, rule onset/center/iou, tolerance, scope)
  2. ranking stability: Kendall's tau between each variant's ranking and the main
     protocol's ranking; how often the main-protocol winner stays first (main excluded)
  3. spread: min-max pooled F1 of each detector across variants, overall, over matching
     rules only (main reference) and over references only (main rule), and after removing
     the shift common to all detectors (per-variant mean); plus a variance decomposition
     of F1 into protocol, detector and protocol x detector parts
  4. uncertainty: subject-level bootstrap 95% CI of pooled F1 under the main protocol, of
     the F1 difference between adjacent detectors, and of every pairwise difference
     (NAP: both naps of a subject are resampled together; DREAMS: one excerpt per subject)
NAP marks sit at the spindle peak, so onset-rule variants are mis-specified there; they are
kept (flagged) to show what that choice does, and the headline NAP statistics are also
given without them.
Writes results/validity/*.csv and a rank heatmap per dataset/event.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from spindlebench import paths, registry
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

MAIN = {"dreams_spindles": ("expert1", "onset", 0.5), "dreams_kc": ("expert1", "onset", 0.5),
        "nap": ("expert", "center", 0.5)}
MAIN_SCOPES = ["N2", "all"]  # N2 = the primary protocol of RQ1
VARIANT = ["reference", "scope", "rule", "tol"]
OUT = paths.RESULTS / "validity"
N_BOOT = 2000


def pooled(g):
    tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan


def bootstrap(g, subject_of, n=N_BOOT, seed=0):
    """subject-level bootstrap of pooled F1 -> per-detector samples (n x detectors)"""
    rng = np.random.default_rng(seed)
    g = g.assign(subject=g.rec.map(subject_of))
    piv = {c: g.pivot_table(index="subject", columns="detector", values=c, aggfunc="sum") for c in ("tp", "fp", "fn")}
    subj = piv["tp"].index.to_numpy()
    idx = rng.integers(0, len(subj), (n, len(subj)))
    tp, fp, fn = (piv[c].to_numpy(float)[idx].sum(1) for c in ("tp", "fp", "fn"))
    with np.errstate(invalid="ignore"):
        return pd.DataFrame(2 * tp / (2 * tp + fp + fn), columns=piv["tp"].columns)


def decompose(f1):
    """two-way decomposition of F1 (variants x detectors): shares of the sum of squares"""
    x = f1.to_numpy(float)
    gm = x.mean()
    rv, cd = x.mean(1, keepdims=True), x.mean(0, keepdims=True)
    ss_var = x.shape[1] * ((rv - gm) ** 2).sum()
    ss_det = x.shape[0] * ((cd - gm) ** 2).sum()
    ss_int = ((x - rv - cd + gm) ** 2).sum()
    tot = ss_var + ss_det + ss_int
    return dict(share_protocol=ss_var / tot, share_detector=ss_det / tot, share_interaction=ss_int / tot)


def analyse(g, ds, ev, main_scope, pool, subject_of):
    dets = sorted(g.detector.unique())
    f1 = g.groupby(VARIANT + ["detector"]).apply(pooled, include_groups=False).unstack("detector")
    f1 = f1.dropna(thresh=len(dets))  # variants where every detector was scored
    off = paths.RESULTS / "benchmark/dreams_official_protocol.csv"
    if ds == "dreams_spindles" and off.exists():  # official DREAMS any-overlap protocol as one more variant
        o = pd.read_csv(off).query("reference == 'union' and excerpts == '1-6'").set_index("detector").f1
        if set(dets) <= set(o.index):
            f1.loc[("union", "all", "any-overlap (official)", 0.0), :] = o[f1.columns].values
    ref, rule, tol = MAIN[ds]
    main = (ref, main_scope, rule, tol)
    if main not in f1.index:
        print(f"skip {ds}/{ev}: main protocol missing")
        return None
    tag = dict(dataset=ds, event=ev, main_scope=main_scope, pool=pool)
    m = f1.loc[main]
    winner = m.idxmax()
    misspec = lambda idx: ds == "nap" and idx[2] == "onset"  # NAP marks are peaks, not onsets

    stab = []
    for idx, row in f1.iterrows():
        stab.append(dict(**tag, **dict(zip(VARIANT, idx)), is_main=idx == main, nap_onset_misspecified=misspec(idx),
                         kendall_tau=kendalltau(m.values, row.values).statistic,
                         winner=row.idxmax(), same_winner=row.idxmax() == winner,
                         **{f"rank_{d}": int(r) for d, r in row.rank(ascending=False, method="min").items()}))
    stab = pd.DataFrame(stab)

    rules = f1[f1.index.get_level_values("reference") == ref]
    refs = f1[(f1.index.get_level_values("rule") == rule) & (f1.index.get_level_values("tol") == tol)]
    centred = f1.sub(f1.mean(1), axis=0)  # shift common to all detectors removed
    spread = pd.DataFrame({**tag, "detector": f1.columns, "in_sample": [d in registry.IN_SAMPLE.get(ds, ()) for d in f1.columns],
                           "F1_main": m.values, "F1_min": f1.min().values, "F1_max": f1.max().values,
                           "F1_range": (f1.max() - f1.min()).values,
                           "F1_range_rules": (rules.max() - rules.min()).values,
                           "F1_range_refs": (refs.max() - refs.min()).values,
                           "F1_range_centred": (centred.max() - centred.min()).values,
                           "rank_main": m.rank(ascending=False, method="min").values,
                           "rank_best": f1.rank(axis=1, ascending=False, method="min").min().values,
                           "rank_worst": f1.rank(axis=1, ascending=False, method="min").max().values})
    top = m.sort_values(ascending=False).index[:9]
    others = stab[~stab.is_main]
    valid = others[~others.nap_onset_misspecified]
    summary = dict(**tag, n_detectors=len(dets), n_variants=len(f1), winner=winner,
                   same_winner=others.same_winner.mean(), same_winner_valid=valid.same_winner.mean(),
                   tau_median=others.kendall_tau.median(), tau_min=others.kendall_tau.min(),
                   median_range=spread.F1_range.median(), median_range_rules=spread.F1_range_rules.median(),
                   median_range_refs=spread.F1_range_refs.median(), median_range_centred=spread.F1_range_centred.median(),
                   top9_spread_main=m[top].max() - m[top].min(),
                   median_top9_spread=(f1[top].max(1) - f1[top].min(1)).median(), **decompose(f1))

    gm = g[(g.reference == ref) & (g.scope == main_scope) & (g.rule == rule) & (g.tol == tol)]
    bs = bootstrap(gm, subject_of)
    order = m.sort_values(ascending=False).index
    ci = [dict(**tag, detector=d, F1=m[d], lo=bs[d].quantile(0.025), hi=bs[d].quantile(0.975)) for d in order]
    for a, b in zip(order[:-1], order[1:]):  # is each step in the ranking resolved?
        diff = bs[a] - bs[b]
        ci.append(dict(**tag, detector=f"{a} - {b}", F1=m[a] - m[b],
                       lo=diff.quantile(0.025), hi=diff.quantile(0.975)))
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            diff = bs[a] - bs[b]
            lo, hi = diff.quantile(0.025), diff.quantile(0.975)
            pairs.append(dict(**tag, a=a, b=b, diff=m[a] - m[b], lo=lo, hi=hi, resolved=lo > 0 or hi < 0))

    print(f"\n=== {ds} / {ev} | main scope {main_scope} | pool {pool}: {len(dets)} detectors, {len(f1)} variants ===")
    print(f"main {main}: winner {winner}; same winner in {summary['same_winner']:.0%} of the other variants "
          f"({summary['same_winner_valid']:.0%} without mis-specified ones); Kendall tau vs main: median "
          f"{summary['tau_median']:.2f}, min {summary['tau_min']:.2f}")
    print(f"median F1 range: all variants {summary['median_range']:.3f}, rules only {summary['median_range_rules']:.3f}, "
          f"references only {summary['median_range_refs']:.3f}, common shift removed {summary['median_range_centred']:.3f}; "
          f"top-9 spread under main {summary['top9_spread_main']:.3f}; SS shares protocol {summary['share_protocol']:.2f} "
          f"detector {summary['share_detector']:.2f} interaction {summary['share_interaction']:.2f}")
    return dict(f1=f1, order=order, stab=stab, spread=spread, summary=summary, ci=pd.DataFrame(ci), pairs=pd.DataFrame(pairs))


def heatmap(res, ds, ev):
    f1, order = res["f1"], res["order"]
    ranks = f1.rank(axis=1, ascending=False, method="min").T.loc[order]
    cols = [f"{r}|{s}|{ru} {t:g}" + (" (mis-spec.)" if ds == "nap" and ru == "onset" else "") for r, s, ru, t in ranks.columns]
    rows = [d + (" *" if d in registry.IN_SAMPLE.get(ds, ()) else "") for d in order]
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(cols)), 0.45 * len(order) + 2.4))
    im = ax.imshow(ranks.values, cmap="Blues_r", aspect="auto", vmin=1, vmax=len(order))
    ax.set_yticks(range(len(order)), rows, fontsize=8)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    ax.set_title(f"{ds} / {ev}: rank of each detector under each protocol variant (1 = best; rows ordered by the "
                 f"N2 main protocol; * = in-sample)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.02, label="rank")
    fig.tight_layout()
    fig.savefig(OUT / f"rank_heatmap_{ds}_{ev}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", default=str(paths.RESULTS / "benchmark/per_recording.csv"))
    ap.add_argument("--min-detectors", type=int, default=3)
    args = ap.parse_args()
    per = pd.read_csv(args.per)
    per = per[~per.excluded & ~per.detector.isin(registry.USES_EXPERT_STAGING)]
    man = pd.read_csv(paths.CACHE / "manifest.csv")
    subject_of = dict(zip(man.dataset + "/" + man.rec, man.subject.astype(str)))
    OUT.mkdir(parents=True, exist_ok=True)
    out = {k: [] for k in ("stab", "spread", "summary", "ci", "pairs")}
    for (ds, ev), g in per.groupby(["dataset", "event"]):
        if ds not in MAIN:
            continue
        # only detectors that cover every recording (agents may still be running)
        cover = g.groupby("detector").rec.nunique()
        partial = sorted(cover[cover < g.rec.nunique()].index)
        if partial:
            print(f"{ds}/{ev}: leaving out partially-run detectors {partial}")
        g = g[~g.detector.isin(partial)]
        sub = {r: subject_of[f"{ds}/{r}"] for r in g.rec.unique()}
        for pool in ("all", "out_of_sample"):
            gp = g if pool == "all" else g[~g.detector.isin(registry.IN_SAMPLE.get(ds, ()))]
            if pool == "out_of_sample" and gp.detector.nunique() == g.detector.nunique():
                continue  # no in-sample detectors here: same as "all"
            if gp.detector.nunique() < args.min_detectors:
                print(f"skip {ds}/{ev}/{pool}: {gp.detector.nunique()} detectors")
                continue
            for scope in MAIN_SCOPES:
                res = analyse(gp, ds, ev, scope, pool, sub.get)
                if res is None:
                    continue
                for k in ("stab", "spread", "ci", "pairs"):
                    out[k].append(res[k])
                out["summary"].append(pd.DataFrame([res["summary"]]))
                if scope == "N2" and pool == "all":
                    heatmap(res, ds, ev)
    if out["stab"]:
        pd.concat(out["stab"]).to_csv(OUT / "rank_stability.csv", index=False)
        pd.concat(out["spread"]).to_csv(OUT / "f1_spread_across_protocols.csv", index=False)
        pd.concat(out["ci"]).to_csv(OUT / "bootstrap_ci_main_protocol.csv", index=False)
        pd.concat(out["pairs"]).to_csv(OUT / "bootstrap_pairwise_main_protocol.csv", index=False)
        summ = pd.concat(out["summary"])
        summ.to_csv(OUT / "validity_summary.csv", index=False)
        pd.set_option("display.width", 250, "display.max_columns", 30)
        print("\n", summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
