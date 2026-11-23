"""
Subject-level splits of the training sites (plan 2.2), written once:

    uv run python splits.py        (from experiments/student)

-> results/student/work/splits.csv: dataset, rec, subject, role, subset
   role    val (10% of subjects per site) | train
   subset  nested data-scale arms for training subjects: "10" ⊂ "30" ⊂ "100" (the smallest arm that
           contains the subject); empty for val. The CPU track uses subset in {"10", "30"}.
One numpy Generator(PCG64, seed 2026) drawn in a fixed order: per site (sorted), a permutation of the
sorted subject ids; the first round(10%) are val, the rest are ranked for the nested subsets.
"""
import numpy as np
import pandas as pd

import data
from spindlebench import provenance

provenance.track(__file__, protocols=[])

VAL_FRAC = 0.10
SUBSETS = (("10", 0.10), ("30", 0.30), ("100", 1.00))


def make(m, seed=data.SEED):
    rng = np.random.default_rng(seed)
    out = []
    for ds in sorted(data.SITES):
        s = m[m.dataset == ds]
        subj = np.array(sorted(s.subject.astype(str).unique(), key=lambda v: (len(v), v)))
        perm = subj[rng.permutation(len(subj))]
        n_val = int(round(VAL_FRAC * len(subj)))
        val, train = perm[:n_val], perm[n_val:]
        tag = {u: "" for u in val}
        for i, u in enumerate(train):
            tag[u] = next(name for name, f in SUBSETS if i < int(round(f * len(train))))
        for _, r in s.iterrows():
            u = str(r.subject)
            out.append(dict(dataset=ds, rec=r.rec, subject=u, role="val" if u in set(val) else "train",
                            subset=tag[u]))
    return pd.DataFrame(out)


def main():
    m = data.manifest()
    sp = make(m)
    data.SPLITS.parent.mkdir(parents=True, exist_ok=True)
    if data.SPLITS.exists():
        old = data.splits()
        if not old.equals(sp.astype(str)):
            raise SystemExit(f"{data.SPLITS} exists and differs from a fresh draw; refusing to overwrite")
        print("splits unchanged")
    sp.to_csv(data.SPLITS, index=False)
    t = sp.assign(n=1).pivot_table(index="dataset", columns=["role", "subset"], values="n", aggfunc="sum")
    print(t.fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
