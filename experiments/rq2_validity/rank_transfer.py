"""
RQ3 at the benchmark level: does a detector ranking established on one dataset carry
over to another? DREAMS spindles (onset ±0.5 s vs expert 1) vs NAP (centre ±0.5 s vs the
expert), pooled F1, detectors run on both.

    uv run python experiments/rq2_validity/rank_transfer.py      # -> results/validity/rank_transfer_*.csv

Detector sets: every detector run on both datasets; out-of-sample (without detectors
whose DREAMS score is in-sample); and family-collapsed (one configuration per algorithm,
spindlebench.registry.FAMILY), out-of-sample. Degenerate settings (F1 <= 0.05 on either dataset,
i.e. almost no detections) are left out. Kendall's tau gets a 95% CI from a subject-level
bootstrap of both datasets at once, and a reliability ceiling: tau between two bootstrap
replicates of the same dataset (how well a ranking agrees with itself under resampling).
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from spindlebench import paths, registry
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

N_BOOT = 2000
OUT = paths.RESULTS / "validity"
per = pd.read_csv(paths.RESULTS / "benchmark/per_recording.csv")
per = per[~per.excluded]
man = pd.read_csv(paths.CACHE / "manifest.csv")
subject_of = dict(zip(man.dataset + "/" + man.rec, man.subject))
PROTO = {"dreams_spindles": ("expert1", "onset"), "nap": ("expert", "center")}


def counts(ds, scope):
    ref, rule = PROTO[ds]
    g = per.query("dataset == @ds and event == 'spindle' and reference == @ref and rule == @rule "
                  "and tol == 0.5 and scope == @scope")
    g = g.assign(subject=(ds + "/" + g.rec).map(subject_of))
    return {c: g.pivot_table(index="subject", columns="detector", values=c, aggfunc="sum") for c in ("tp", "fp", "fn")}


def f1(c, dets, idx=None):
    tp, fp, fn = (c[k][dets].to_numpy(float) for k in ("tp", "fp", "fn"))
    if idx is not None:
        tp, fp, fn = tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1)
    else:
        tp, fp, fn = tp.sum(0), fp.sum(0), fn.sum(0)
    return 2 * tp / (2 * tp + fp + fn)


def tau_rows(x, y):
    return np.array([kendalltau(a, b).statistic for a, b in zip(x, y)])


rng = np.random.default_rng(0)
rows, tables = [], []
for scope in ("N2", "all"):
    cd, cn = counts("dreams_spindles", scope), counts("nap", scope)
    both = sorted(set(cd["tp"].columns) & set(cn["tp"].columns) - registry.USES_EXPERT_STAGING)
    j = pd.DataFrame({"F1_dreams": f1(cd, both), "F1_nap": f1(cn, both)}, index=both)
    j = j[j.min(axis=1) > 0.05]
    sets = {"all": list(j.index),
            "out_of_sample": [d for d in j.index if d not in registry.IN_SAMPLE["dreams_spindles"]]}
    sets["family_collapsed"] = [d for d in sets["out_of_sample"] if registry.FAMILY.get(d, d) == d]
    for name, dets in sets.items():
        t = j.loc[dets].copy()
        t["rank_dreams"] = t.F1_dreams.rank(ascending=False, method="min").astype(int)
        t["rank_nap"] = t.F1_nap.rank(ascending=False, method="min").astype(int)
        t["scope"], t["set"] = scope, name
        tables.append(t.rename_axis("detector").reset_index().sort_values("rank_dreams"))
        # subject-level bootstrap of both datasets; reliability = tau between two replicates of one dataset
        nd, nn = len(cd["tp"]), len(cn["tp"])
        bd = [rng.integers(0, nd, (N_BOOT, nd)) for _ in range(2)]
        bn = [rng.integers(0, nn, (N_BOOT, nn)) for _ in range(2)]
        fd, fn_ = [f1(cd, dets, b) for b in bd], [f1(cn, dets, b) for b in bn]
        tb = tau_rows(fd[0], fn_[0])
        rel_d, rel_n = tau_rows(fd[0], fd[1]), tau_rows(fn_[0], fn_[1])
        dw, nw = t.F1_dreams.idxmax(), t.F1_nap.idxmax()
        rows.append(dict(scope=scope, set=name, n_detectors=len(t),
                         kendall_tau=kendalltau(t.F1_dreams, t.F1_nap).statistic,
                         tau_lo=np.nanpercentile(tb, 2.5), tau_hi=np.nanpercentile(tb, 97.5),
                         tau_p=kendalltau(t.F1_dreams, t.F1_nap).pvalue,
                         reliability_dreams=np.nanmedian(rel_d), reliability_nap=np.nanmedian(rel_n),
                         spearman=spearmanr(t.F1_dreams, t.F1_nap).statistic,
                         dreams_winner=dw, its_nap_rank=int(t.loc[dw, "rank_nap"]),
                         nap_winner=nw, its_dreams_rank=int(t.loc[nw, "rank_dreams"])))
OUT.mkdir(parents=True, exist_ok=True)
pd.concat(tables).to_csv(OUT / "rank_transfer_dreams_nap.csv", index=False)
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "rank_transfer_summary.csv", index=False)
pd.set_option("display.width", 250)
print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
