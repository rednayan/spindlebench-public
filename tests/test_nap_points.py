"""Hand-computed cases for point-mark scoring (protocol nap_points.v1)."""
import numpy as np
import pytest

from spindlebench.protocols.nap_points import KEYS, TAUS, integrated, random_dets, score_nap, summarise, tensors


def test_tolerance_grid_is_fixed():
    np.testing.assert_allclose(TAUS, [0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    assert KEYS[-1] == "contain"


def test_centre_tolerance_sweep():
    # one detection [10, 11] (centre 10.5) and a mark at 11.2: 0.7 s from the centre
    out = score_nap(np.array([10.0]), np.array([11.0]), np.array([11.2]), 60.0)
    assert out["tau0.5"] == (0, 1, 1)
    assert out["tau0.75"] == (1, 0, 0)
    assert out["tau1.5"] == (1, 0, 0)
    # containment widens the detection by 0.25 s: 11.2 lies within [9.75, 11.25]
    assert out["contain"] == (1, 0, 0)


def test_no_detections_or_no_marks():
    out = score_nap(np.zeros(0), np.zeros(0), np.array([5.0]), 60.0)
    assert out["tau0.5"] == (0, 0, 1)
    assert np.isnan(out["aff"][0]) and out["aff"][1] == 0.0
    out = score_nap(np.array([1.0]), np.array([2.0]), np.zeros(0), 60.0)
    assert out["tau0.5"] == (0, 1, 0)
    assert np.isnan(out["aff"][0]) and np.isnan(out["aff"][1])


def test_perfect_detection_has_full_affiliation_recall():
    marks = np.array([10.0, 30.0])
    out = score_nap(marks - 0.5, marks + 0.5, marks, 60.0)
    assert out["aff"][1] == pytest.approx(1.0)
    assert out["tau0.25"] == (2, 0, 0)


def test_integrated_f1_of_a_flat_curve():
    assert integrated(np.full(len(TAUS), 0.4)) == pytest.approx(0.4)
    # linear F1(tau) from 0 to 1 integrates to its mean, 0.5
    assert integrated((TAUS - TAUS[0]) / (TAUS[-1] - TAUS[0])) == pytest.approx(0.5)


def test_pooled_counts_and_bootstrap_shapes():
    sc = {k: (2, 1, 1) for k in KEYS}
    sc["aff"] = (0.5, 0.8)
    C, aff = tensors([("s1", sc), ("s2", sc), ("s1", sc)], ["s1", "s2"])
    assert C[0, 0].tolist() == [4, 2, 2]
    F, integ, P, R, AF = summarise(C, aff)
    assert F[0, 0] == pytest.approx(2 * 6 / (2 * 6 + 3 + 3))
    assert P[0] == pytest.approx(0.5) and R[0] == pytest.approx(0.8)
    W = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    F, integ, *_ = summarise(C, aff, W)
    assert F.shape == (3, len(KEYS)) and integ.shape == (3,)


def test_random_baseline_keeps_count_and_durations():
    info = {"epoch": 30.0, "n2_epochs": np.arange(10)}
    on, off = np.array([5.0, 50.0, 100.0]), np.array([6.0, 51.5, 100.8])
    ro, rf = random_dets(on, off, info, np.random.default_rng(0))
    assert len(ro) == 3
    np.testing.assert_allclose(np.sort(rf - ro), np.sort(off - on))
    assert (ro >= 0).all() and ((ro + rf) / 2 < 300).all()
