"""
SpindleInject generator: spindle injection into real EEG background.

Pure numpy/scipy/pandas so it runs in the main environment and in the detector environments
(the SUMOv2 training code imports this file directly, without the spindlebench package).
Everything is in raw uV at the recording's native rate, before any detector preprocessing.

Parameters are external and fixed in advance (AASM/MODA band, literature durations). None are
fitted to expert labels.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

try:
    from spindlebench import paths as _paths
    ROOT, CACHE, RESULTS = _paths.ROOT, _paths.CACHE, _paths.RESULTS
except ImportError:  # imported as a plain file from an environment without the package
    ROOT = Path(__file__).resolve().parents[3]
    CACHE, RESULTS = ROOT / "cache", ROOT / "results"

# detectors whose detections define "not quiet"; the ones present for a dataset are used
QUIET_DETECTORS = ["sumov2", "sumo", "a7", "moelle2011_faithful", "wonambi_moelle2011",
                   "yasa_dreams_tuned", "yasa_default", "openspindlenet"]


@dataclass(frozen=True)
class Params:
    freq: tuple = (11.0, 16.0)        # carrier start frequency, uniform (Hz)
    drift: tuple = (-1.0, 0.5)        # linear frequency change over the event, uniform (Hz)
    dur_median: float = 0.8           # log-normal duration (s)
    dur_sigma: float = 0.35           # of log duration
    dur_clip: tuple = (0.5, 2.0)
    snr_db: tuple = (-3.0, 12.0)      # uniform; RMS(spindle) / RMS(background sigma band, +-5 s)
    rate_per_min: float = 6.0         # target events per minute of eligible (quiet N2) signal
    min_gap: float = 3.0              # s between event edges
    quiet_margin: float = 1.0         # s kept free around any existing detection
    snr_win: float = 5.0              # half-window for background sigma RMS (s)


DEFAULT_PARAMS = Params()  # frozen, so one shared default is safe


def sigma_band(x, sf, lo=11.0, hi=16.0):
    sos = butter(4, [lo, hi], btype="band", fs=sf, output="sos")
    return sosfiltfilt(sos, x)


def spindle(sf, dur, f0, drift, rng):
    """unit-RMS spindle waveform: chirped cosine under a waxing-waning envelope"""
    n = max(int(round(dur * sf)), 4)
    t = np.arange(n) / sf
    f = f0 + drift * t / dur
    phase = 2 * np.pi * np.cumsum(f) / sf + rng.uniform(0, 2 * np.pi)
    if rng.random() < 0.5:
        env = tukey(n, alpha=rng.uniform(0.6, 1.0))
    else:
        env = np.exp(-0.5 * ((t - dur / 2) / (dur / rng.uniform(4.0, 6.0))) ** 2)
    # mild asymmetry: the peak sits between 35% and 65% of the event
    skew = rng.uniform(0.35, 0.65)
    warp = np.interp(np.linspace(0, 1, n), [0, skew, 1], [0, 0.5, 1])
    env = np.interp(warp, np.linspace(0, 1, n), env)
    w = env * np.cos(phase)
    return w / np.sqrt(np.mean(w ** 2))


def stage_mask(dataset, rec, n, sf, stages=("N2",)):
    h = pd.read_csv(CACHE / "labels" / dataset / f"{rec}__hypno.csv")
    if len(h) < 2:
        return np.zeros(n, bool)
    epoch = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    idx = np.clip((np.arange(n) / sf // epoch).astype(int), 0, len(h) - 1)
    m = np.isin(h.stage.to_numpy()[idx], stages)
    m[np.arange(n) / sf >= h.epoch_start.iloc[-1] + epoch] = False   # past the last scored epoch
    return m


def busy_mask(dataset, rec, channel, n, sf, margin, detectors=QUIET_DETECTORS):
    """True where any available detector fired (+- margin). Returns (mask, detectors used)."""
    m = np.zeros(n, bool)
    used = []
    for d in detectors:
        f = RESULTS / "detections" / dataset / d / f"{rec}.csv"
        if not f.exists():
            continue
        used.append(d)
        ev = pd.read_csv(f)
        if "channel" in ev and ev.channel.notna().any():
            ev = ev[ev.channel.astype(str) == channel]
        for on, du in zip(ev.onset.to_numpy(float), ev.duration.fillna(0).to_numpy(float)):
            a, b = int((on - margin) * sf), int((on + du + margin) * sf) + 1
            m[max(a, 0):min(b, n)] = True
    return m, used


def inject(x, sf, eligible, params=DEFAULT_PARAMS, seed=0, snr_db=None):
    """
    x: raw signal (uV); eligible: bool mask where an event may lie entirely.
    snr_db: None -> drawn from params.snr_db; a scalar or an array -> cycled (benchmark grid).
    Returns (x + spindles, events DataFrame).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = len(x)
    sig = sigma_band(x, sf)
    out = x.copy()
    free = eligible.copy()
    n_target = int(round(eligible.sum() / sf / 60 * params.rate_per_min))
    gap = int(params.min_gap * sf)
    rows, tries = [], 0
    grid = None if snr_db is None else np.atleast_1d(np.asarray(snr_db, float))
    while len(rows) < n_target and tries < 50 * max(n_target, 1):
        tries += 1
        dur = float(np.clip(np.exp(np.log(params.dur_median) + params.dur_sigma * rng.standard_normal()),
                            *params.dur_clip))
        L = int(round(dur * sf))
        cand = np.flatnonzero(free[: n - L])
        if not len(cand):
            break
        a = int(rng.choice(cand))
        if not free[a:a + L].all():
            continue
        f0 = rng.uniform(*params.freq)
        dr = rng.uniform(*params.drift)
        snr = rng.uniform(*params.snr_db) if grid is None else grid[len(rows) % len(grid)]
        w0, w1 = max(a - int(params.snr_win * sf), 0), min(a + L + int(params.snr_win * sf), n)
        bg = np.sqrt(np.mean(sig[w0:w1] ** 2))
        w = spindle(sf, dur, f0, dr, rng) * bg * 10 ** (snr / 20)
        out[a:a + L] += w[:L]
        free[max(a - gap, 0):a + L + gap] = False
        rows.append(dict(onset=a / sf, duration=L / sf, freq=f0, drift=dr, snr_db=snr,
                         amp_ptp=float(w.max() - w.min()), bg_sigma_rms=bg))
    ev = pd.DataFrame(rows, columns=["onset", "duration", "freq", "drift", "snr_db", "amp_ptp", "bg_sigma_rms"])
    return out, ev.sort_values("onset").reset_index(drop=True)


def eligible_mask(dataset, rec, channel, n, sf, params=DEFAULT_PARAMS, stages=("N2",)):
    st = stage_mask(dataset, rec, n, sf, stages)
    busy, used = busy_mask(dataset, rec, channel, n, sf, params.quiet_margin)
    return st & ~busy, st, used


def params_dict(params=DEFAULT_PARAMS):
    return asdict(params)
