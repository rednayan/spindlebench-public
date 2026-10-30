# SPDX-FileCopyrightText: original authors (see below); 2026 Nayan Sharma (Python port)
# SPDX-License-Identifier: GPL-3.0-or-later
# Port of GPL-3.0 code (see below); distributed under the GPL, unlike the rest of this repository (MIT).
"""
Python port of the official A7 spindle detector (MATLAB, v1.1.2)
    Lacourse, Delfrate, Beaudry, Peppard & Warby (2019) J Neurosci Methods 316:3-11
    https://github.com/swarby/A7_LacourseSpindleDetector  (commit e67b6372, GPL-3.0)

Line-by-line port of lib/a7SpindleDetection.m and its sub-functions with the
defaults of a7MainScript.m / initA7_DEF.m. Differences to the MATLAB code are
listed in README.md (filter edge padding, one known index bug not replicated).
Everything works on 0-based sample indices; MATLAB 1-based positions are kept
where the original logic depends on them (baseline window selection).
"""
import math
import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt

DEF = dict(
    fs=100,                                   # DEF_a7.standard_sampleRate
    absSigPow_Th=1.25, relSigPow_Th=1.6, sigCov_Th=1.3, sigCorr_Th=0.69,
    minDurSpindleSec=0.3, maxDurSpindleSec=2.5,
    lowFreqLow=0.5, lowFreqHigh=8.0, highFreqLow=16.0, highFreqHigh=30.0, slowRat_Th=0.9,
    sigmaFreqLow=11.0, sigmaFreqHigh=16.0, fOrder=20,
    totalFreqLow=4.5, totalFreqHigh=30.0,
    winLengthSec=0.3, WinStepSec=0.1, ZeroPadSec=2, bslLengthSec=30,
    lowPerct=10, highPerct=90,
)


# ---------------------------------------------------------------- helpers
def mround(x):
    """MATLAB round (half away from zero)"""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def sigma_filter(x, d):
    """butterFiltZPHighPassFiltFilt(11 Hz) then butterFiltZPLowPassFiltFilt(16 Hz), order 20, zero phase"""
    hp = butter(d["fOrder"], d["sigmaFreqLow"], "highpass", fs=d["fs"], output="sos")
    lp = butter(d["fOrder"], d["sigmaFreqHigh"], "lowpass", fs=d["fs"], output="sos")
    return sosfiltfilt(lp, sosfiltfilt(hp, x))


def samples2windows(x, n_win, d):
    """samples2WindowsInSec: (n_win, win_len) matrix, incomplete windows NaN-padded"""
    wl = mround(d["winLengthSec"] * d["fs"])
    out = np.full((n_win, wl), np.nan)
    n = len(x)
    for i in range(n_win):
        s = mround(i * d["WinStepSec"] * d["fs"])
        if s >= n:
            s = n - 1
        e = min(s + wl, n)
        out[i, :e - s] = x[s:e]
    return out


def windows2samples_mean(vals, n, d, reduce="mean"):
    """windows2SamplesInSec followed by mean(...,'omitnan') (or max) over overlapping windows,
    then fillmissing(...,'previous')"""
    wl = mround(d["winLengthSec"] * d["fs"])
    step = d["WinStepSec"] * d["fs"]
    sup = math.ceil(d["winLengthSec"] / d["WinStepSec"])
    if n > mround(len(vals) * step + (wl - step)):
        raise ValueError("data per window are missing to generate the data per sample")
    mat = np.full((sup, n), np.nan)
    for i, v in enumerate(vals):
        s = mround(i * step)
        if s >= n:
            continue
        mat[i % sup, s:min(s + wl, n)] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmax(mat, 0) if reduce == "max" else np.nanmean(mat, 0)
    # fillmissing 'previous'
    idx = np.where(~np.isnan(out), np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = out[idx]
    filled[np.isnan(out) & (np.cumsum(~np.isnan(out)) == 0)] = np.nan
    return filled


def matlab_prctile(x, p):
    """percentiles.m (MATLAB prctile): midpoint interpolation == numpy 'hazen'"""
    return np.percentile(x, p, method="hazen")


def find_freq_index(lo, hi, freqs):
    """findFreqIndex.m, returned as a python slice (inclusive stop)"""
    def nearest(f):
        i = np.nonzero(freqs >= f)[0]
        i = i[0] if len(i) else len(freqs) - 1
        if i > 0 and abs(freqs[i - 1] - f) < abs(freqs[i] - f):
            i -= 1
        return i
    return slice(nearest(lo), nearest(hi) + 1)


def n_full_windows(n, d):
    return int(math.floor((n / d["fs"] - d["winLengthSec"]) / d["WinStepSec"])) + 1


def n_round_windows(n, d):
    return mround((n / d["fs"]) / d["WinStepSec"])


def valid_windows(valid_samples, n_win, d):
    inv = samples2windows((~valid_samples).astype(float), n_win, d)
    return np.nansum(inv, 1) == 0


def baseline_slices(valid_by_win, d):
    """For every window, the slice of positions in iAvailable used as its baseline
    (the 30-s baseline loop shared by a7subRelSigPow / a7subSigmaCov / a7subTurnOffDetSlowRatio)."""
    n_win = len(valid_by_win)
    avail = np.nonzero(valid_by_win)[0]            # 0-based window indices
    L = len(avail)
    nb = int(math.floor((d["bslLengthSec"] - d["winLengthSec"]) / d["WinStepSec"])) + 1
    if L == 0:
        raise ValueError("Bsl is empty, no detection possible")
    nb = min(nb, L)
    starts = np.empty(n_win, int)
    for w in range(n_win):                         # MATLAB iPSDWinTot = w + 1
        pos = np.searchsorted(avail, w, "left")    # first position with avail >= w
        # MATLAB: iValPSDWin = find(iAvailable>=iPSDWinTot,1,'first')  -> 1-based position
        #         if empty: iValPSDWin = iAvailable(end)  (a window index, kept as in the original)
        ival = pos + 1 if pos < L else avail[-1] + 1
        if ival <= math.ceil((nb - 1) / 2):
            starts[w] = 0
        elif ival >= L - math.floor((nb - 1) / 2):
            starts[w] = L - nb
        else:
            starts[w] = ival - math.ceil((nb - 1) / 2)
    return avail, starts, nb


# ---------------------------------------------------------------- features
def abs_sigma_pow(x, sig, d):
    n = len(x)
    wl = mround(d["winLengthSec"] * d["fs"])
    W = samples2windows(sig, n_round_windows(n, d), d)
    e = np.nansum(W ** 2 / wl, 1)
    return np.log10(windows2samples_mean(e, n, d))


def psd_windows(x, d):
    """calcPSDNormType(..., 'hann', removeMean=1, zeroPad=1, 'IntegSpectPow')"""
    fs = d["fs"]
    nfft = mround(d["ZeroPadSec"] * fs)
    wl = mround(d["winLengthSec"] * fs)
    nw = n_full_windows(len(x), d)
    nbins = nfft // 2 + 1 if nfft % 2 == 0 else (nfft + 1) // 2
    freqs = np.arange(nbins) * fs / nfft
    wt = np.arange(wl) / (wl - 1)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * wt)
    ng = np.sum(hann ** 2) / wl
    starts = np.array([mround(i * d["WinStepSec"] * fs) for i in range(nw)])
    W = x[starts[:, None] + np.arange(wl)[None, :]]
    mu = W.mean(1)
    W = (W - mu[:, None]) * hann
    nz = mround(nfft - wl)
    Z = np.zeros((nw, nfft))
    o = mround(nz / 2)
    Z[:, o:o + wl] = W
    mod2 = (np.abs(np.fft.fft(Z, nfft, axis=1)[:, :nbins]) / nfft) ** 2
    mod2[:, 1:] *= 2
    psd = mod2 / ng
    psd[:, 0] = mu
    return psd, freqs


def rel_sigma_pow(x, valid_samples, d):
    n = len(x)
    psd, freqs = psd_windows(x, d)
    sig = psd[:, find_freq_index(d["sigmaFreqLow"], d["sigmaFreqHigh"], freqs)].sum(1)
    tot = psd[:, find_freq_index(d["totalFreqLow"], d["totalFreqHigh"], freqs)].sum(1)
    low = psd[:, find_freq_index(d["lowFreqLow"], d["lowFreqHigh"], freqs)].sum(1)
    high = psd[:, find_freq_index(d["highFreqLow"], d["highFreqHigh"], freqs)].sum(1)
    with np.errstate(all="ignore"):
        rel = np.log10(sig / tot)
    vbw = valid_windows(valid_samples, len(rel), d)
    bsl = np.where(vbw, rel, np.nan)
    z = _rescore(rel, bsl, vbw, d)
    if mround(len(z) * d["WinStepSec"] * d["fs"] + (d["winLengthSec"] - d["WinStepSec"]) * d["fs"]) < n:
        z = np.r_[z, 0.0]
    return windows2samples_mean(z, n, d), low, high


def _rescore(values_all, values_bsl, vbw, d):
    """baseline stats from the valid windows (values_bsl), applied to values_all"""
    avail, starts, nb = baseline_slices(vbw, d)
    v = values_bsl[avail]
    lim = matlab_prctile(v[~np.isnan(v)], [d["lowPerct"], d["highPerct"]])
    keep = (v >= lim[0]) & (v <= lim[1])
    vk = np.where(keep, v, 0.0)
    c1, s1, s2 = (np.r_[0, np.cumsum(a)] for a in (keep, vk, vk * vk))
    a, b = starts, starts + nb
    cnt = (c1[b] - c1[a]).astype(float)
    sm, sq = s1[b] - s1[a], s2[b] - s2[a]
    with np.errstate(all="ignore"):
        mean = sm / cnt
        sd = np.sqrt(np.maximum((sq - sm * sm / cnt) / (cnt - 1), 0))
    sd[cnt == 1] = 0.0
    mean, sd = np.nan_to_num(mean, nan=0.0), np.nan_to_num(sd, nan=0.0)
    with np.errstate(all="ignore"):
        return (values_all - mean) / sd


def sigma_cov(x, sig, valid_samples, d):
    n = len(x)
    nw = n_full_windows(n, d)
    R = samples2windows(x, nw, d)
    S = samples2windows(sig, nw, d)
    Rc = R - R.mean(1, keepdims=True)
    Sc = S - S.mean(1, keepdims=True)
    cov = (Rc * Sc).sum(1) / (R.shape[1] - 1)
    cov[cov <= 0] = 0
    lcov = np.log10(cov + 1)
    vbw = valid_windows(valid_samples, nw, d)
    z = _rescore(lcov, lcov, vbw, d)
    if mround(nw * d["WinStepSec"] * d["fs"] + (d["winLengthSec"] * d["fs"] - d["WinStepSec"] * d["fs"])) < n:
        z = np.r_[z, 0.0]
    return windows2samples_mean(z, n, d)


def sigma_corr(x, sig, d):
    n = len(x)
    nw = n_round_windows(n, d)
    R = samples2windows(x, nw, d)
    S = samples2windows(sig, nw, d)
    Rc = R - R.mean(1, keepdims=True)          # NaN rows (incomplete windows) stay NaN, as corrcoef in MATLAB
    Sc = S - S.mean(1, keepdims=True)
    with np.errstate(all="ignore"):
        r = (Rc * Sc).sum(1) / np.sqrt((Rc ** 2).sum(1) * (Sc ** 2).sum(1))
    if mround(nw * d["WinStepSec"] * d["fs"] + (d["winLengthSec"] * d["fs"] - d["WinStepSec"] * d["fs"])) < n:
        r = np.r_[r, 0.0]
    return windows2samples_mean(r, n, d)


def slow_ratio_valid(low, high, artifact, n, d):
    """a7subTurnOffDetSlowRatio: per-sample IN-context flag"""
    nw = n_full_windows(n, d)
    vbw = valid_windows(~artifact, nw, d)
    avail, starts, nb = baseline_slices(vbw, d)
    lo, hi = low[avail], high[avail]
    cl, ch = np.r_[0, np.cumsum(lo)], np.r_[0, np.cumsum(hi)]
    a, b = starts, starts + nb
    ratio = ((cl[b] - cl[a]) / nb) / ((ch[b] - ch[a]) / nb)
    ratio[ratio <= 0] = 0
    ratio = np.log10(ratio + 1)
    valid = (ratio > d["slowRat_Th"]).astype(float)
    if mround(nw * d["WinStepSec"] * d["fs"] + (d["winLengthSec"] - d["WinStepSec"]) * d["fs"]) < n:
        valid = np.r_[valid, 0.0]
    return windows2samples_mean(valid, n, d, reduce="max") > 0


def runs(mask):
    """event_StartsEndsDurations: inclusive (start, end) sample indices of True runs"""
    m = np.r_[False, mask.astype(bool), False]
    dm = np.diff(m.astype(int))
    return np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1


# ---------------------------------------------------------------- main
def a7_detect(x, stage_valid=None, artifact=None, d=DEF):
    """
    x            EEG at d['fs'] (100 Hz), µV, broadband 0.3-30 Hz
    stage_valid  bool per sample, True where the sleep stage belongs to the baseline (N2)
                 (None -> every sample is baseline, bslSleepStaging empty)
    artifact     bool per sample (None -> no artifacts)
    returns      list of (start_sample, end_sample_inclusive, in_context), features (n, 4)
    """
    x = np.asarray(x, float)
    n = len(x)
    artifact = np.zeros(n, bool) if artifact is None else np.asarray(artifact, bool)
    stage_valid = np.ones(n, bool) if stage_valid is None else np.asarray(stage_valid, bool)
    valid = stage_valid & ~artifact

    sig = sigma_filter(x, d)
    f_abs = abs_sigma_pow(x, sig, d)
    f_rel, low, high = rel_sigma_pow(x, valid, d)
    f_cov = sigma_cov(x, sig, valid, d)
    f_cor = sigma_corr(x, sig, d)
    feats = np.c_[f_abs, f_rel, f_cov, f_cor]

    with np.errstate(invalid="ignore"):
        p_abs = f_abs > d["absSigPow_Th"]
        p_rel = f_rel > d["relSigPow_Th"]
        p_cov = f_cov > d["sigCov_Th"]
        p_cor = f_cor > d["sigCorr_Th"]
    poss = p_abs & p_rel & p_cov & p_cor
    starts, ends = runs(poss)
    plen = p_abs & p_cov
    # extend each event while absSigPow and sigmaCov stay above threshold
    if len(starts):
        rs, re_ = runs(plen)                     # every poss sample lies inside a plen run
        run_id = np.searchsorted(rs, starts, "right") - 1
        starts, ends = rs[run_id], re_[run_id]
        se = np.unique(np.c_[starts, ends], axis=0)
        starts, ends = se[:, 0], se[:, 1]
    dur = ends - starts + 1
    keep = (dur >= d["minDurSpindleSec"] * d["fs"]) & (dur <= d["maxDurSpindleSec"] * d["fs"])
    det = np.zeros(n, bool)
    for s, e in zip(starts[keep], ends[keep]):
        det[s:e + 1] = True
    # organiseOutputTxtFile re-extracts the events from the detection vector
    starts, ends = runs(det)
    ctx = slow_ratio_valid(low, high, artifact, n, d)
    events = [(int(s), int(e), bool(ctx[s:e + 1].any())) for s, e in zip(starts, ends)]
    return events, feats
