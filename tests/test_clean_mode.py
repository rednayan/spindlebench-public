"""detectors/common.py --clean: events found on joined clean windows map back to recording time."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("common", Path(__file__).resolve().parents[1] / "detectors" / "common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

SF = 100.0


def ev(onset, duration):
    return dict(onset=onset, duration=duration, channel="C3", event="spindle", score=None)


def test_no_mask_is_identity():
    idx = np.arange(int(60 * SF))
    out = common.to_recording_time([ev(12.34, 0.8)], idx, SF)
    assert np.isclose(out[0]["onset"], 12.34) and np.isclose(out[0]["duration"], 0.8)


def test_events_after_a_gap_are_shifted_back():
    keep = np.ones(int(60 * SF), bool)
    keep[int(20 * SF):int(25 * SF)] = False  # one flagged 5-s window at 20-25 s
    idx = np.flatnonzero(keep)
    out = common.to_recording_time([ev(30.0, 1.0), ev(5.0, 1.0)], idx, SF)
    assert [round(o["onset"], 2) for o in out] == [35.0, 5.0]
    assert all(np.isclose(o["duration"], 1.0) for o in out)


def test_events_at_or_near_a_join_are_dropped():
    keep = np.ones(int(60 * SF), bool)
    keep[int(20 * SF):int(25 * SF)] = False
    idx = np.flatnonzero(keep)  # join at clean time 20 s
    crossing, near, far = ev(19.5, 1.0), ev(20.5, 0.6), ev(22.0, 0.8)
    out = common.to_recording_time([crossing, near, far], idx, SF)
    assert [round(o["onset"], 2) for o in out] == [27.0]
