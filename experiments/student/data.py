"""
Shared definitions for the student study (notes/plans/student_architecture.md): paths, sites,
teachers, the 100 Hz grid, the loss mask, and the gold-set guard.

Step 1 uses: guard, splits, preprocess, valid_mask, events_to_mask, mask_to_events, load_*.
Crops and augmentation (section 2.5, 4.6) are added in step 2.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.data.cache import labels_dir, signal_path

SITES = ("hmc", "cinc2018")            # the only datasets training, validation and selection may read
GOLD = ("dreams_spindles", "nap")      # read only by label_models.py --gold (teacher outputs) and infer.py
TEACHERS = ("a7", "luna_spindles", "wonambi_ucsd", "yasa_default")
FS = 100                               # model grid (Hz), as SUMOv2
SEED = 2026
JOIN_GUARD = 1.0                       # = detectors/common.py: teacher events this close to a flagged window were dropped

STUDY = paths.RESULTS / "student"
WORK = STUDY / "work"
TEACHER_STORE = paths.RESULTS / "detections_clean"   # teachers run on artefact-free windows (artefacts.v1)
SPLITS = WORK / "splits.csv"
# training runs (one folder per run) live outside the repository: SPINDLEBENCH_RUNS, default <root>/../runs/student
RUNS = Path(os.environ.get("SPINDLEBENCH_RUNS") or paths.ROOT.parent / "runs" / "student").resolve()


def run_name(arm, tag, seed, track="cpu"):
    """folder name of one run: <arm>[_<tag>][_<track>]_s<seed>; the CPU track (the first grid) has no suffix"""
    return arm + (f"_{tag}" if tag else "") + ("" if track == "cpu" else f"_{track}") + f"_s{seed}"


class GoldSetError(RuntimeError):
    pass


def guard(dataset):
    """training, validation and selection code calls this for every recording it opens"""
    if dataset not in SITES:
        raise GoldSetError(f"{dataset!r} is not a training site {SITES}; gold sets are read by infer.py only")
    return dataset


def manifest(datasets=SITES):
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    return m[m.dataset.isin(datasets)].reset_index(drop=True)


def splits():
    return pd.read_csv(SPLITS, dtype=str, keep_default_na=False)


# ------------------------------------------------------------------ signal

def _sumo_adapter():
    spec = importlib.util.spec_from_file_location("sumo_run", paths.ROOT / "detectors" / "sumo" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SUMO = None


def sumo():
    """detectors/sumo/run.py (preprocess_sumov2, merge_and_filter_spindles), loaded once"""
    global _SUMO
    if _SUMO is None:
        _SUMO = _sumo_adapter()
    return _SUMO


def load_signal(row):
    """default channel of one recording, native rate, float64"""
    z = np.load(signal_path(row.path, paths.CACHE / "manifest.csv"))
    ch = [str(c) for c in z["ch_names"]]
    return z["data"][ch.index(str(row.default_channel))].astype(float), float(z["sf"])


def preprocess(x, sf):
    """SUMOv2 inference preprocessing, unchanged (plan 2.4) -> float32 at 100 Hz"""
    return sumo().preprocess_sumov2(x, sf).astype(np.float32)


# ------------------------------------------------------------------ masks on the 100 Hz grid

def stage_per_sample(dataset, rec, n):
    """stage label of each 100 Hz sample from the public / expert hypnogram"""
    h = pd.read_csv(labels_dir() / dataset / f"{rec}__hypno.csv")
    epoch = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    idx = np.clip((np.arange(n) / FS // epoch).astype(int), 0, len(h) - 1)
    return h.stage.to_numpy().astype(str)[idx]


def clean_per_sample(dataset, rec, n):
    """False inside flagged artefact windows and within JOIN_GUARD s of them (where the teachers,
    run with --clean, could not report events)"""
    a = pd.read_csv(labels_dir() / dataset / f"{rec}__artefact.csv")
    ok = np.ones(n, bool)
    for s, e in a.loc[a.artefact == 1, ["start", "end"]].to_numpy():
        ok[max(int(round((s - JOIN_GUARD) * FS)), 0):int(round((e + JOIN_GUARD) * FS))] = False
    return ok


def valid_mask(dataset, rec, n):
    """samples that enter any loss or label model: N2 of the public hypnogram and clean"""
    return (stage_per_sample(dataset, rec, n) == "N2") & clean_per_sample(dataset, rec, n)


def events_to_mask(onset, duration, n):
    """binary 100 Hz mask, 1 on [round(onset*FS), round((onset+duration)*FS))"""
    y = np.zeros(n, np.uint8)
    for o, d in zip(np.asarray(onset, float), np.nan_to_num(np.asarray(duration, float))):
        i0, i1 = int(round(o * FS)), int(round((o + d) * FS))
        y[max(i0, 0):max(min(i1, n), 0)] = 1
    return y


def mask_to_events(y):
    """(start, stop) sample indices of the runs of 1 in a binary vector"""
    d = np.diff(np.r_[0, np.asarray(y, np.int8), 0])
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1)]


def load_teacher_events(dataset, rec, teacher, channel):
    d = pd.read_csv(TEACHER_STORE / dataset / teacher / f"{rec}.csv", dtype={"channel": str})
    d = d[(d.event == "spindle") & (d.channel == str(channel))]
    return d.onset.to_numpy(float), d.duration.to_numpy(float)


def n_samples(row):
    """length of the 100 Hz grid for a recording (= len(resample_poly(x, 100, sf)))"""
    z = np.load(signal_path(row.path, paths.CACHE / "manifest.csv"))
    n_native = z["data"].shape[-1]
    return int(np.ceil(n_native * FS / float(z["sf"])))


def prep_path(dataset, rec):
    return WORK / "prep" / dataset / f"{rec}.npz"


def teacher_path(dataset, rec):
    return WORK / "teachers" / dataset / f"{rec}.npz"


def load_teachers(dataset, rec):
    """(Y uint8 [n_teachers, n], valid bool [n]) for a training-site recording"""
    guard(dataset)
    z = np.load(teacher_path(dataset, rec))
    return np.stack([z[t] for t in TEACHERS]), z["valid"].astype(bool)


def here():
    return Path(__file__).resolve().parent


# ------------------------------------------------------------------ training arrays (memory-mapped)

def x_path(dataset, rec):
    return WORK / "arrays" / dataset / f"{rec}__x.npy"


def code_path(dataset, rec):
    return WORK / "arrays" / dataset / f"{rec}__code.npy"


VALID_BIT = 1 << len(TEACHERS)   # code = sum_i 2^i y_i  (+16 if valid)


def ensure_arrays(dataset, rec):
    """float32 signal and uint8 code per night as .npy (np.load(mmap_mode="r") shares pages between jobs)"""
    guard(dataset)
    xp, cp = x_path(dataset, rec), code_path(dataset, rec)
    if not xp.exists():
        xp.parent.mkdir(parents=True, exist_ok=True)
        np.save(xp, np.load(prep_path(dataset, rec))["x"].astype(np.float32))
    if not cp.exists():
        Y, valid = load_teachers(dataset, rec)
        code = (Y.astype(np.uint8) << np.arange(len(TEACHERS), dtype=np.uint8)[:, None]).sum(0).astype(np.uint8)
        np.save(cp, (code | (valid.astype(np.uint8) * VALID_BIT)).astype(np.uint8))
    return xp, cp


def open_night(dataset, rec):
    xp, cp = ensure_arrays(dataset, rec)
    return np.load(xp, mmap_mode="r"), np.load(cp, mmap_mode="r")


def open_witness(dataset, rec):
    """memory-mapped uint8 multi-channel witness mask from prepare_witness.py"""
    guard(dataset)
    p = WORK / "witness" / dataset / f"{rec}__mc.npy"
    if not p.exists():
        raise FileNotFoundError(f"missing witness mask {p}; run `python prepare_witness.py` first")
    return np.load(p, mmap_mode="r")


CROP, CORE = 4000, 3000          # 40-s crop, loss on the central 30 s
CORE_LO = (CROP - CORE) // 2


class CropSampler:
    """40-s crops with the start uniform over every position at which the crop lies wholly inside a
    contiguous run of valid samples (N2 and clean) >= 40 s (plan 2.5). Shorter runs are never sampled;
    their count and total length are kept in short_runs / short_s."""

    def __init__(self, recs, use_witness=False):
        self.x, self.code, self.mc = [], [], []
        night, start, count = [], [], []
        self.short_runs, self.short_s, self.valid_s = 0, 0.0, 0.0
        for k, (ds, rec) in enumerate(recs):
            x, code = open_night(ds, rec)
            self.x.append(x)
            self.code.append(code)
            if use_witness:
                self.mc.append(open_witness(ds, rec))
            v = (np.asarray(code) & VALID_BIT) > 0
            self.valid_s += v.sum() / FS
            d = np.diff(np.r_[0, v.astype(np.int8), 0])
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if b - a >= CROP:
                    night.append(k)
                    start.append(a)
                    count.append(b - a - CROP + 1)
                else:
                    self.short_runs += 1
                    self.short_s += (b - a) / FS
        self.night, self.start = np.array(night), np.array(start)
        self.cum = np.r_[0, np.cumsum(count)]
        self.positions = int(self.cum[-1])
        self.sampled_s = float((np.array(count) + CROP - 1).sum() / FS)

    def crops_per_pass(self):
        """one pass = every crop position once in expectation: sampled valid seconds / 30 s"""
        return int(self.sampled_s // (CORE / FS))

    def batch_with_nights(self, rng, n, return_mc=False):
        """-> x [n, CROP] float32, code [n, CROP] uint8, night_idx [n] int (, mc [n, CROP] uint8)"""
        u = rng.integers(0, self.positions, n)
        r = np.searchsorted(self.cum, u, side="right") - 1
        st = self.start[r] + (u - self.cum[r])
        nights = self.night[r]
        x = np.stack([self.x[k][a:a + CROP] for k, a in zip(nights, st)]).astype(np.float32)
        c = np.stack([self.code[k][a:a + CROP] for k, a in zip(nights, st)])
        if return_mc:
            mc = np.stack([self.mc[k][a:a + CROP] for k, a in zip(nights, st)])
            return x, c, nights, mc
        return x, c, nights

    def batch(self, rng, n):
        """-> x [n, CROP] float32, code [n, CROP] uint8"""
        x, c, _ = self.batch_with_nights(rng, n)
        return x, c


QUIET_TILE = 1000     # 10 s teacher-free N2 tile
QUIET_FADE = 25       # 0.25 s Hann cross-fade between stitched tiles


class QuietN2TileSampler:
    """Samples seamless 40-s quiet N2 crops (all 4 teachers == 0) from the same night by stitching
    10-s completely teacher-free N2 tiles with 0.25-s cross-fades. Falls back to any training night
    if a night has no 10-s quiet run."""

    def __init__(self, crop_sampler):
        self.x = crop_sampler.x
        self.per_night = []
        all_n, all_s, all_c = [], [], []
        tile_len = QUIET_TILE + QUIET_FADE
        for k, code in enumerate(crop_sampler.code):
            c = np.asarray(code)
            q = (c == VALID_BIT)      # valid N2 & clean, and all 4 teacher bits == 0
            d = np.diff(np.r_[0, q.astype(np.int8), 0])
            ns, nc = [], []
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if b - a >= tile_len:
                    ns.append(a)
                    nc.append(b - a - tile_len + 1)
                    all_n.append(k)
                    all_s.append(a)
                    all_c.append(b - a - tile_len + 1)
            if ns:
                cum = np.r_[0, np.cumsum(nc)]
                self.per_night.append((np.array(ns), cum, int(cum[-1])))
            else:
                self.per_night.append(None)
        self.all_n = np.array(all_n)
        self.all_s = np.array(all_s)
        self.all_cum = np.r_[0, np.cumsum(all_c)]
        self.all_pos = int(self.all_cum[-1])
        self._fade_in = (0.5 * (1.0 - np.cos(np.pi * np.arange(QUIET_FADE) / QUIET_FADE))).astype(np.float32)
        self._fade_out = 1.0 - self._fade_in

    def _one_tile(self, k, rng):
        L = QUIET_TILE + QUIET_FADE
        entry = self.per_night[int(k)]
        if entry is not None:
            starts, cum, pos = entry
            u = int(rng.integers(0, pos))
            r = int(np.searchsorted(cum, u, side="right") - 1)
            a = int(starts[r] + (u - cum[r]))
            return np.asarray(self.x[int(k)][a:a + L], dtype=np.float32)
        u = int(rng.integers(0, self.all_pos))
        r = int(np.searchsorted(self.all_cum, u, side="right") - 1)
        a = int(self.all_s[r] + (u - self.all_cum[r]))
        return np.asarray(self.x[int(self.all_n[r])][a:a + L], dtype=np.float32)

    def sample_for_nights(self, night_indices, rng):
        """-> [len(night_indices), CROP] float32 seamless quiet N2 background from the same night"""
        n_tiles = (CROP + QUIET_TILE - 1) // QUIET_TILE
        out = np.empty((len(night_indices), CROP), dtype=np.float32)
        for i, k in enumerate(night_indices):
            buf = np.zeros(n_tiles * QUIET_TILE + QUIET_FADE, dtype=np.float32)
            for j in range(n_tiles):
                t = self._one_tile(k, rng)
                pos = j * QUIET_TILE
                if j == 0:
                    buf[pos:pos + QUIET_TILE + QUIET_FADE] = t
                else:
                    buf[pos:pos + QUIET_FADE] = (buf[pos:pos + QUIET_FADE] * self._fade_out
                                                 + t[:QUIET_FADE] * self._fade_in)
                    buf[pos + QUIET_FADE:pos + QUIET_TILE + QUIET_FADE] = t[QUIET_FADE:]
            out[i] = buf[:CROP]
        return out


def mix_faint(x, quiet_x, rng, prob=0.5, alpha_lo=0.45, alpha_hi=0.75):
    """Same-night quiet-N2 mixing on a random `prob` fraction of crops:
    x_mix = alpha * x + sqrt(1 - alpha^2) * quiet_x, preserving variance and dropping spindle SNR ~2.5-7 dB."""
    out = x.copy()
    mask = rng.random(len(x)) < prob
    if mask.any():
        a = rng.uniform(alpha_lo, alpha_hi, int(mask.sum())).astype(np.float32)[:, None]
        b = np.sqrt(1.0 - a ** 2)
        out[mask] = a * out[mask] + b * quiet_x[mask]
    return out, mask


def phase_randomize_batch(x, rng):
    """Batch FFT phase-randomised surrogates of 1-D crops [K, T]: exact same power spectrum (incl. N2
    sigma power and slow waves) with zero coherent transient bursts (hard within-N2 negatives, y = 0)."""
    n = x.shape[-1]
    X = np.fft.rfft(np.asarray(x, dtype=np.float64), axis=-1)
    hi = X.shape[-1] - 1 if n % 2 == 0 else X.shape[-1]
    if hi > 1:
        phi = rng.uniform(0.0, 2.0 * np.pi, size=(len(x), hi - 1))
        X[:, 1:hi] *= np.exp(1j * phi)
    return np.fft.irfft(X, n=n, axis=-1).astype(np.float32)


class WakeRemSampler:
    """40-s clean Wake or REM crops (certain background, y = 0) from the training nights."""

    def __init__(self, recs):
        self.x = []
        night, start, count = [], [], []
        self.valid_s = 0.0
        for k, (ds, rec) in enumerate(recs):
            x, _ = open_night(ds, rec)
            self.x.append(x)
            n = len(x)
            st = stage_per_sample(ds, rec, n)
            ok = ((st == "W") | (st == "REM")) & clean_per_sample(ds, rec, n)
            self.valid_s += ok.sum() / FS
            d = np.diff(np.r_[0, ok.astype(np.int8), 0])
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if b - a >= CROP:
                    night.append(k)
                    start.append(a)
                    count.append(b - a - CROP + 1)
        self.night, self.start = np.array(night), np.array(start)
        self.cum = np.r_[0, np.cumsum(count)]
        self.positions = int(self.cum[-1]) if len(count) else 0

    def batch(self, rng, n):
        """-> x [n, CROP] float32"""
        u = rng.integers(0, self.positions, n)
        r = np.searchsorted(self.cum, u, side="right") - 1
        st = self.start[r] + (u - self.cum[r])
        return np.stack([self.x[k][a:a + CROP] for k, a in zip(self.night[r], st)]).astype(np.float32)


def augment(x, rng):
    """SUMOv2's amplitude scaling (factor uniform in [0.5, 1] or [1, 2], each with probability 1/2),
    sign flip with probability 1/2, Gaussian noise at 1% of each crop's SD. No time reversal."""
    n = len(x)
    down = rng.random(n) < 0.5
    f = np.where(down, rng.uniform(0.5, 1.0, n), rng.uniform(1.0, 2.0, n))
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    x = x * (f * sign)[:, None]
    return (x + 0.01 * x.std(1, keepdims=True) * rng.standard_normal(x.shape)).astype(np.float32)


def decode_code(code):
    """uint8 code -> (Y [.., R, T] teacher bits, valid [.., T] bool, pattern [.., T] 0..15)"""
    pat = code & (VALID_BIT - 1)
    Y = np.stack([(pat >> i) & 1 for i in range(len(TEACHERS))], axis=-2)
    return Y, (code & VALID_BIT) > 0, pat
