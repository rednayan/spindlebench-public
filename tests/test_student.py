"""experiments/student, step 1: gold-set guard, masks, splits, matching, label-model tables."""
import sys

import numpy as np
import pandas as pd
import pytest

from spindlebench import paths
from spindlebench.metrics.events import match

sys.path.insert(0, str(paths.ROOT / "experiments" / "student"))
import data  # noqa: E402
import label_models  # noqa: E402
import rasterise  # noqa: E402
import splits  # noqa: E402


def test_guard_refuses_gold_sets():
    for ds in data.GOLD:
        with pytest.raises(data.GoldSetError):
            data.guard(ds)
    assert data.guard("hmc") == "hmc"


def test_mask_round_trip():
    y = data.events_to_mask([1.0, 3.205], [0.5, 1.0], 600)
    ev = data.mask_to_events(y)
    assert ev.tolist() == [[100, 150], [320, 420]]
    assert data.mask_to_events(np.zeros(5)).shape == (0, 2)


def test_clean_mask_leaves_out_the_join_guard(tmp_path, monkeypatch):
    (tmp_path / "x").mkdir()
    pd.DataFrame(dict(start=[0, 5, 10, 15], end=[5, 10, 15, 20], stage="N2", zscore=0.0, artefact=[0, 1, 0, 0])) \
        .to_csv(tmp_path / "x" / "r__artefact.csv", index=False)
    monkeypatch.setattr(data, "labels_dir", lambda: tmp_path)
    ok = data.clean_per_sample("x", "r", 2000)
    bad = np.flatnonzero(~ok)
    assert bad[0] == 400 and bad[-1] == 1099  # [5 - 1, 10 + 1) s


def test_cluster_matching_equals_full_matching():
    rng = np.random.default_rng(0)
    for _ in range(30):
        a, b = np.sort(rng.uniform(0, 60, 40)), np.sort(rng.uniform(0, 60, 35))
        full = match(np.c_[a, 0 * a], np.c_[b, 0 * b], "onset", 0.5)[0]
        assert rasterise.match_tp(a, b) == full


def test_splits_are_subject_disjoint_nested_and_reproducible():
    m = pd.DataFrame(dict(dataset=["hmc"] * 50 + ["cinc2018"] * 80, rec=[f"r{i}" for i in range(130)],
                          subject=[i // 2 for i in range(50)] + list(range(100, 180))))
    a, b = splits.make(m), splits.make(m)
    assert a.equals(b)
    by = a.groupby("subject").role.nunique()
    assert (by == 1).all()  # a subject's nights never straddle train and val
    tr = a[a.role == "train"]
    for _, s in tr.groupby("dataset"):
        n = s.subject.nunique()
        n10 = s[s.subset == "10"].subject.nunique()
        n30 = s[s.subset.isin(["10", "30"])].subject.nunique()
        assert n10 == round(0.1 * n) and n30 == round(0.3 * n)
    assert (a[a.role == "val"].subset == "").all()


def test_label_model_tables():
    P = label_models.PAT
    assert P.shape == (16, 4) and P[5].tolist() == [1, 0, 1, 0]  # bit i = TEACHERS[i]
    counts = np.arange(16) + 1.0
    ids = label_models.sample_patterns(counts, n=100_000)
    share = np.bincount(ids, minlength=16) / len(ids)
    assert np.allclose(share, counts / counts.sum(), atol=0.005)


def test_gold_decoder_follows_d0():
    post = np.zeros(16)
    post[15] = 1.0
    code = np.zeros(1000, int)
    code[100:150] = 15   # 0.5 s: kept
    code[300:310] = 15   # 0.1 s: dropped (< 0.3 s)
    code[500:800] = 15   # 3.0 s: dropped (> 2.5 s)
    ev, _ = label_models.decode(post, code)
    assert np.allclose(ev, [[1.0, 1.5]])


def test_exact_ds_recovers_known_teachers():
    P = label_models.PAT
    pi, sens, spec = 0.05, np.array([0.8, 0.6, 0.7, 0.5]), np.array([0.99, 0.998, 0.98, 0.995])
    l1 = pi * np.prod(np.where(P, sens, 1 - sens), 1)
    l0 = (1 - pi) * np.prod(np.where(P, 1 - spec, spec), 1)
    q, info = label_models.ds_exact((l1 + l0) * 1e9)
    assert np.allclose(info["recall"], sens, atol=0.02) and abs(info["prior_spindle"] - pi) < 0.005
    assert np.allclose(q, l1 / (l1 + l0), atol=0.02)
