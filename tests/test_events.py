"""Hand-computed cases for event matching and P/R/F1."""
import numpy as np
import pytest

from spindlebench.metrics.events import clip, in_mask, intersection, iou_matrix, match, prf, union


def ev(*rows):
    return np.array(rows, float).reshape(-1, 2)


def test_onset_tolerance_is_inclusive():
    # 0.5 s apart counts despite float rounding (0.1 + 0.4 != 0.5 exactly)
    tp, fp, fn, _ = match(ev([10.1 + 0.4, 1]), ev([10.0, 1]), "onset", 0.5)
    assert (tp, fp, fn) == (1, 0, 0)
    tp, fp, fn, _ = match(ev([10.51, 1]), ev([10.0, 1]), "onset", 0.5)
    assert (tp, fp, fn) == (0, 1, 1)


def test_matching_is_one_to_one():
    # two detections near one reference: one TP, one FP
    tp, fp, fn, pairs = match(ev([10.0, 1], [10.2, 1]), ev([10.1, 1]), "onset", 0.5)
    assert (tp, fp, fn) == (1, 1, 0)
    assert len(pairs) == 1


def test_matching_maximises_number_of_pairs():
    # greedy nearest-first would pair det 0 (10.4) with ref 1 (10.5) and leave ref 0 (9.95) unmatched;
    # the maximum assignment pairs det 0 with ref 0 and det 1 with ref 1
    det = ev([10.4, 1], [10.9, 1])
    ref = ev([9.95, 1], [10.5, 1])
    tp, fp, fn, pairs = match(det, ref, "onset", 0.5)
    assert (tp, fp, fn) == (2, 0, 0)
    assert sorted(pairs) == [(0, 0), (1, 1)]


def test_iou_matrix_values():
    m = iou_matrix(ev([0, 2]), ev([1, 2], [5, 1]))
    assert m[0, 0] == pytest.approx(1 / 3)  # overlap 1 s, union 3 s
    assert m[0, 1] == 0.0


def test_iou_rule_threshold():
    det, ref = ev([0, 2]), ev([1, 2])  # IoU 1/3
    assert match(det, ref, "iou", 0.2)[:3] == (1, 0, 0)
    assert match(det, ref, "iou", 0.4)[:3] == (0, 1, 1)


def test_empty_inputs():
    assert match(ev(), ev([1, 1]), "onset")[:3] == (0, 0, 1)
    assert match(ev([1, 1]), ev(), "iou")[:3] == (0, 1, 0)


def test_prf():
    assert prf(3, 1, 2) == pytest.approx((0.75, 0.6, 6 / 9))
    p, r, f = prf(0, 0, 0)
    assert np.isnan(p) and np.isnan(r) and np.isnan(f)


def test_union_and_intersection_of_experts():
    a = ev([10, 1], [20, 1])
    b = ev([10.5, 1], [30, 1])
    np.testing.assert_array_equal(union(a, b), ev([10, 1], [20, 1], [30, 1]))
    np.testing.assert_array_equal(intersection(a, b), ev([10, 1]))


def test_clip_and_mask():
    e = ev([5, 1], [999, 1], [1000, 1])
    np.testing.assert_array_equal(clip(e, 0, 1000), ev([5, 1], [999, 1]))
    # midpoint rule; a NaN duration is a point mark
    e = ev([9.6, 1], [11, np.nan])
    np.testing.assert_array_equal(in_mask(e, lambda t: t >= 10), e)
    np.testing.assert_array_equal(in_mask(e, lambda t: t > 10.5), e[1:])
