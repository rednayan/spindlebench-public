"""YASA spindle detection settings and a one-channel wrapper (used by detectors/yasa and the sweeps)."""

from spindlebench.metrics.events import empty

YASA_DEFAULT = {"rel_pow": 0.2, "corr": 0.65, "rms": 1.5}
YASA_LOOSE = {"rel_pow": 0.15, "corr": 0.55, "rms": 1.0}
# threshold sweep shared by every dataset. Wide enough that the NAP optimum is interior
# (the earlier 0.05-0.30 / 0.40-0.75 / 0.75-2.0 grid put it on a corner); rel_pow 0 disables
# the relative-power criterion
THRESH_GRID = {
    "rel_pow": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "corr": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "rms": [0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00],
}


def run_yasa(sig_uv, sf, thresh, hypno=None, include=(2,)):
    """YASA spindle detection on one channel (µV). Broadband capped below Nyquist."""
    import yasa

    broad = (1, min(30, 0.45 * sf))
    sp = yasa.spindles_detect(sig_uv, sf, thresh=thresh, freq_broad=broad,
                              hypno=hypno, include=include, verbose="ERROR")
    if sp is None:
        return empty()
    return sp.summary()[["Start", "Duration"]].to_numpy()
