"""
Step 1: preprocess every included NAP nap (C3) and the 8 DREAMS spindle excerpts exactly like the
zero-shot adapter, build the +-0.4 s target masks, and write the subject-grouped fold assignment.

    detectors/sumo/.venv-sumo/Scripts/python.exe experiments/rq4_adaptation/sumo_finetune/prepare.py

Folds (seed 0): the 17 included subjects are shuffled and split into 5 test folds (4,4,3,3,3
subjects). Within each fold's 13-14 non-test subjects, 2 are drawn as validation subjects (early
stopping + threshold of the fine-tuned models); the rest are training subjects in a fixed random
order, so the learning-curve subsets (first 1 / 3 / 6 / all) are nested. Excluded naps never appear.
"""
import numpy as np
import pandas as pd

import ftlib as L

SEED = 0
N_FOLDS = 5
N_VAL = 2


def main():
    m = L.manifest()
    inc = L.nap_included()
    for row in pd.concat([inc, m[m.dataset == "dreams_spindles"]]).itertuples():
        L.prepare_recording(row)
        print("prepared", row.dataset, row.rec, flush=True)

    rng = np.random.default_rng(SEED)
    subjects = np.array(sorted(inc.subject.unique()))
    perm = rng.permutation(subjects)
    test_groups = np.array_split(perm, N_FOLDS)
    rows = []
    for k, test in enumerate(test_groups):
        rest = rng.permutation(np.setdiff1d(subjects, test))
        val, train = rest[:N_VAL], rest[N_VAL:]
        for s in test:
            rows.append(dict(fold=k, subject=s, role="test", train_rank=-1))
        for s in val:
            rows.append(dict(fold=k, subject=s, role="val", train_rank=-1))
        for i, s in enumerate(train):
            rows.append(dict(fold=k, subject=s, role="train", train_rank=i))
    f = pd.DataFrame(rows)
    recs = inc.groupby("subject").rec.apply(lambda r: ";".join(sorted(r)))
    f["recs"] = f.subject.map(recs)
    L.WORK.mkdir(parents=True, exist_ok=True)
    f.to_csv(L.WORK / "folds.csv", index=False)
    # sanity: every included subject is test exactly once, no excluded nap anywhere
    t = f[f.role == "test"]
    assert sorted(t.subject) == sorted(subjects) and t.subject.is_unique
    excl = set(m[(m.dataset == "nap") & m.excluded.astype(bool)].rec)
    assert not excl & set(";".join(f.recs).split(";"))
    print(f.to_string(index=False))


if __name__ == "__main__":
    main()
