"""
Step 5: tables with paired bootstrap CIs for the SUMOv2 injection experiment (main venv).

    uv run python experiments/injection/sumo_inject_ft/report.py

Primary (fixed in docs/thesis/plan_injection.md): NAP pooled N2 F1, detection centre vs mark +-0.5 s,
C (sumov2_inj) minus A0 (sumov2_zeroshot); subject-level paired bootstrap (both naps of a person
together; 2000 resamples, seed 0). Each trained arm is the mean over its 3 seeds: in every resample
the pooled F1 of each seed is computed and the 3 are averaged.
Secondary: DREAMS (onset +-0.5 s vs expert 1, 0-1000 s, N2; excerpt = person), in-domain
leave-one-excerpt-out and NAP-trained (`_x`).
-> results/injection/benchmark/{nap_table.csv, dreams_table.csv, contrasts.csv}
"""
from pathlib import Path

import numpy as np
import pandas as pd
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

ROOT = Path(__file__).resolve().parents[3]
BM = ROOT / "results" / "injection" / "benchmark"
ARMS = {"A0 zero-shot": ["sumov2_zeroshot"], "A1 recal (real)": ["sumov2_recal"],
        "A1' recal (injected)": ["sumov2_injrecal"],
        "B self-training": ["sumov2_self", "sumov2_self_s1", "sumov2_self_s2"],
        "C injection": ["sumov2_inj", "sumov2_inj_s1", "sumov2_inj_s2"],
        "C' injection, real-label selection": ["sumov2_inj_rv", "sumov2_inj_rv_s1", "sumov2_inj_rv_s2"],
        "D real labels": ["sumov2_real", "sumov2_real_s1", "sumov2_real_s2"]}
CONTRASTS = [("C injection", "A0 zero-shot"), ("C injection", "B self-training"), ("C injection", "D real labels"),
             ("D real labels", "A0 zero-shot"), ("B self-training", "A0 zero-shot"),
             ("C' injection, real-label selection", "A0 zero-shot"), ("A1' recal (injected)", "A0 zero-shot")]
PROT = {"nap": dict(reference="expert", scope="N2", rule="center", tol=0.5),
        "dreams_spindles": dict(reference="expert1", scope="N2", rule="onset", tol=0.5)}


def f1(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)


def load(ds, suffix=""):
    pr = pd.read_csv(BM / "per_recording_partial.csv")
    for k, v in PROT[ds].items():
        pr = pr[pr[k] == v]
    pr = pr[(pr.dataset == ds) & ~pr.excluded.astype(bool)]
    unit = (lambda r: r.split("_")[0]) if ds == "nap" else str
    recs = sorted(pr.rec.unique())
    units = sorted({unit(r) for r in recs})
    ui = np.array([units.index(unit(r)) for r in recs])
    C = {}
    for arm, ids in ARMS.items():
        ids = [i + (suffix if not i.endswith(("zeroshot", "recal")) else "") for i in ids]
        mats = []
        for i in ids:
            g = pr[pr.detector == i].set_index("rec").reindex(recs)
            if g.tp.isna().any():
                break
            mats.append(g[["tp", "fp", "fn"]].to_numpy(float))
        else:
            # per unit counts, shape (n_seeds, n_units, 3)
            C[arm] = np.stack([np.stack([m[ui == u].sum(0) for u in range(len(units))]) for m in mats])
    return C, len(units)


def table(ds, suffix="", B=2000):
    C, nu = load(ds, suffix)
    rng = np.random.default_rng(0)
    W = rng.multinomial(nu, np.full(nu, 1 / nu), size=B).astype(float)
    point = {a: np.nanmean(f1(c.sum(1))) for a, c in C.items()}
    boot = {a: np.nanmean(f1(np.einsum("bu,suk->bsk", W, c)), axis=1) for a, c in C.items()}
    rows = []
    for a, c in C.items():
        tot = c.sum(1)
        rows.append(dict(dataset=ds + suffix, arm=a, seeds=len(c), F1=point[a], F1_seed_min=np.nanmin(f1(tot)),
                         F1_seed_max=np.nanmax(f1(tot)), P=np.mean(tot[:, 0] / (tot[:, 0] + tot[:, 1])),
                         R=np.mean(tot[:, 0] / (tot[:, 0] + tot[:, 2]))))
    con = []
    for a, b in CONTRASTS:
        if a in C and b in C:
            d = boot[a] - boot[b]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            con.append(dict(dataset=ds + suffix, contrast=f"{a} - {b}", diff=point[a] - point[b], lo=lo, hi=hi,
                            excludes_0=bool(lo > 0 or hi < 0), units=nu))
    if "C injection" in C and "D real labels" in C:
        gap = boot["D real labels"] - boot["A0 zero-shot"]
        frac = (boot["C injection"] - boot["A0 zero-shot"]) / np.where(np.abs(gap) > 1e-6, gap, np.nan)
        lo, hi = np.nanpercentile(frac, [2.5, 97.5])
        con.append(dict(dataset=ds + suffix, contrast="fraction of A0->D gap closed by C",
                        diff=(point["C injection"] - point["A0 zero-shot"]) / (point["D real labels"] - point["A0 zero-shot"]),
                        lo=lo, hi=hi, excludes_0=bool(lo > 0 or hi < 0), units=nu))
    return pd.DataFrame(rows), pd.DataFrame(con)


def main():
    t, c = [], []
    for ds, suf in [("nap", ""), ("dreams_spindles", ""), ("dreams_spindles", "_x")]:
        a, b = table(ds, suf)
        t.append(a), c.append(b)
    t, c = pd.concat(t), pd.concat(c)
    t.round(4).to_csv(BM / "arms_table.csv", index=False)
    c.round(4).to_csv(BM / "contrasts.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
