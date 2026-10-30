"""
Confirmation tests for the DREAMS assessment method (protocols.dreams_assessment), which was written
from the published description of Devuyst et al. (2011), not from the DREAMS MATLAB code.

1. Rules from the description, on hand-made events.
2. The original: counts from a local run of the DREAMS MATLAB assessment (MATLAB R2025b, official
   script, duration_event = 1) for the detections shipped with DREAMS, and the published result
   (sensitivity 70.20%, false positives 26.44% of reference events).
3. The earlier literal Python port of the MATLAB code (retired, commit 79ab311): its table for every
   detector, kept in tests/data/dreams_official_reference.csv. The two agree on 107 of 140 rows; the rest
   differ by at most 4 events. The differences were traced by running the old port as a black box on
   single events:
     - a detection starting at exactly 0 s is counted twice by the old port (an indexing slip at the
       first sample): A7 (excerpt 4) and SSSM (excerpt 2);
     - clusters in which the detector and an expert are linked only through the other expert, or touch
       without sharing a sample at the end of the recording, which the published description does not
       cover (wonambi FASST2 and Ray 2015, SSSM, OpenSpindleNet).
"""
import numpy as np
import pandas as pd
import pytest

from spindlebench import paths
from spindlebench.protocols import dreams_assessment as da

SF = 100
N = 100 * SF


def mask(*events):
    return da.to_mask(np.array(events, float).reshape(-1, 2), SF, N)


# ------------------------------------------------------------------ rules from the description

def test_overlap_is_agreement_and_touching_is_not():
    v1 = mask([10, 1])                      # 10.00-11.00 s
    assert da.count(mask([10.5, 1]), v1, None, "expert1") == (1, 0, 0)
    # starts where the expert event ends: one cluster, no shared sample -> one outcome, an FP
    assert da.count(mask([11, 1]), v1, None, "expert1") == (0, 1, 0)


def test_each_cluster_counts_once():
    v1 = mask([10, 2])                      # one expert event
    det = mask([10.2, 0.3], [11.0, 0.5])    # two detections inside it
    assert da.count(det, v1, None, "expert1") == (1, 0, 0)
    # a detection bridging two expert events also makes one cluster
    assert da.count(mask([10.5, 2]), mask([10, 1], [12, 1]), None, "expert1") == (1, 0, 0)


def test_one_scorer_touching_events_are_separated():
    m = mask([10, 1], [11, 1])
    assert len(da.clusters(m)) == 2


def test_separate_events_give_fp_and_fn():
    assert da.count(mask([10, 1]), mask([50, 1]), None, "expert1") == (0, 1, 1)


def test_union_and_intersection():
    v1, v2 = mask([10, 1], [30, 1]), mask([10.5, 1], [60, 1])
    det = mask([10.8, 0.5], [30.2, 0.3])
    assert da.count(det, v1, v2, "union") == (2, 0, 1)           # 10-11.5 (both), 30 (V1); 60 missed
    assert da.count(det, v1, v2, "intersection") == (1, 1, 0)    # only 10-11.5 has both experts


def test_measures():
    m = da.measures(377, 142, 160, duration=6 * 1800, duration_event=1)
    assert m["sens"] == pytest.approx(377 / 537) and m["fp_prop"] == pytest.approx(142 / 537)
    assert m["tn"] == 6 * 1800 - 377 - 142 - 160


# ------------------------------------------------------------------ the original and the published result

MATLAB = {  # (excerpts, reference): TP, FP, FN of the DREAMS automatic detections, official MATLAB script
    ("1-6", "expert1"): (197, 322, 91),
    ("1-6", "expert2"): (318, 201, 90),
    ("1-6", "union"): (377, 142, 160),
    ("1-6", "intersection"): (138, 381, 21),
    ("1-8", "expert1"): (237, 446, 117),
}


@pytest.fixture(scope="module")
def table():
    if not (paths.CACHE / "manifest.csv").exists():
        pytest.skip("no cache (DREAMS is downloaded by the user)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("dreams_official",
                                                  paths.ROOT / "experiments/rq1_benchmark/dreams_official.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.table()


@pytest.mark.data
def test_matches_the_original_matlab_counts(table):
    t = table[table.detector == "dreams_auto"].set_index(["excerpts", "reference"])
    for key, counts in MATLAB.items():
        assert tuple(t.loc[key, ["TP", "FP", "FN"]]) == counts, key


@pytest.mark.data
def test_reproduces_the_published_result(table):
    r = table[(table.detector == "dreams_auto") & (table.excerpts == "1-6") & (table.reference == "union")].iloc[0]
    assert round(100 * r.sens, 2) == 70.20 and round(100 * r.fp_prop, 2) == 26.44


@pytest.mark.data
def test_agrees_with_the_earlier_port(table):
    ref = pd.read_csv(paths.ROOT / "tests" / "data" / "dreams_official_reference.csv")
    m = ref.merge(table, on=["detector", "excerpts", "reference"], suffixes=("_old", "_new"))
    assert len(m) == len(ref)
    d = np.abs(m[["TP_old", "FP_old", "FN_old"]].to_numpy() - m[["TP_new", "FP_new", "FN_new"]].to_numpy())
    assert (d.sum(1) == 0).sum() >= 107
    assert d.max() <= 4
    assert (m.f1_old - m.f1_new).abs().max() < 0.003
    differing = set(m.detector[d.sum(1) > 0])
    assert differing <= {"a7", "a7_n2base", "sssm", "sssm_v2", "openspindlenet", "wonambi_fasst2", "wonambi_ray2015"}
