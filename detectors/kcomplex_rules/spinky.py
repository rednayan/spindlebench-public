# SPDX-FileCopyrightText: original authors (see below); 2026 Nayan Sharma (Python port)
# SPDX-License-Identifier: GPL-3.0-or-later
# Port of GPL-3.0 code (see below); distributed under the GPL, unlike the rest of this repository (MIT).
"""
Python port of the K-complex part of Spinky (Lajnef et al. 2017, Front Neuroinform 11:15)
    https://github.com/TarekLaj/SPINKY  (commit a2cb3293, GPL-3.0)
which bundles Selesnick's TQWT toolbox (tqwt_radix2 / itqwt_radix2 / dualQ / ComputeNow).

Pipeline per 30-s epoch (functions/test_process.m -> signal_decomposition.m -> kc_detection.m):
  1. zero-phase FIR band-pass 0.2-45 Hz (fir1 order 105, filtfilt)
  2. MCA: dualQ(x, Q1=5.5, r1=3, J1=MaxLevels, Q2=1, r2=3, J2=MaxLevels, theta=0.5, mu=0.005, Nit=500)
     -> oscillatory (high-Q) and transient (low-Q) components; K-complexes come from the transient one
  3. transient - mean, causal FIR band-pass 0.8-4 Hz (fir1 order 100, filter)
  4. troughs at least 1.5 s apart whose value <= threshold (µV), more than 1 s into the epoch,
     and deeper than 2.5x the filtered value 1 s earlier
The TQWT is computed with precomputed frequency-domain gather maps (each radix-2 TQWT subband is
a re-indexed, re-weighted slice of the signal's DFT), vectorised over all epochs of a recording;
tests in check_spinky.py verify it against the literal afb/sfb/lps cascade.
"""
import math

import numpy as np
import scipy.fft as sfft
from scipy import sparse
from scipy.signal import find_peaks, filtfilt, firwin, lfilter


# ------------------------------------------------------------ literal TQWT pieces (MATLAB toolbox)
def _trans(T):
    v = np.arange(1, T + 1) / (T + 1) * np.pi
    return (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2


def afb_map(idx, w, N0, N1):
    """afb.m on a 'gather' representation: spectrum element k = X0[idx[k]] * w[k]"""
    N = len(idx)
    P, T, S = (N - N1) // 2, (N0 + N1 - N) // 2 - 1, (N - N0) // 2
    tr = _trans(T)
    i0, w0 = np.zeros(N0, int), np.zeros(N0)
    i0[0], w0[0] = idx[0], w[0]
    k = np.arange(1, P + 1)
    i0[k], w0[k] = idx[k], w[k]
    k = np.arange(1, T + 1)
    i0[P + k], w0[P + k] = idx[P + k], w[P + k] * tr
    w0[N0 // 2] = 0
    i0[N0 - P - k], w0[N0 - P - k] = idx[N - P - k], w[N - P - k] * tr
    k = np.arange(1, P + 1)
    i0[N0 - k], w0[N0 - k] = idx[N - k], w[N - k]

    i1, w1 = np.zeros(N1, int), np.zeros(N1)
    k = np.arange(1, T + 1)
    i1[k], w1[k] = idx[P + k], w[P + k] * tr[T - k]
    k = np.arange(1, S + 1)
    i1[T + k], w1[T + k] = idx[P + T + k], w[P + T + k]
    if N % 2 == 0:
        i1[N1 // 2], w1[N1 // 2] = idx[N // 2], w[N // 2]
    i1[N1 - T - k], w1[N1 - T - k] = idx[N - P - T - k], w[N - P - T - k]
    k = np.arange(1, T + 1)
    i1[N1 - k], w1[N1 - k] = idx[N - P - k], w[N - P - k] * tr[T - k]
    return (i0, w0), (i1, w1)


def lps_map(idx, w, N0):
    """lps.m (low-pass scaling) on the gather representation"""
    N = len(idx)
    i, ww = np.zeros(N0, int), np.zeros(N0)
    if N0 <= N:
        k = np.arange(0, N0 // 2)
        i[k], ww[k] = idx[k], w[k]
        i[N0 // 2], ww[N0 // 2] = idx[N // 2], w[N // 2]
        k = np.arange(1, N0 // 2)
        i[N0 - k], ww[N0 - k] = idx[N - k], w[N - k]
    else:
        k = np.arange(0, N // 2)
        i[k], ww[k] = idx[k], w[k]
        i[N0 // 2], ww[N0 // 2] = idx[N // 2], w[N // 2]
        k = np.arange(1, N // 2)
        i[N0 - k], ww[N0 - k] = idx[N - k], w[N - k]
    return i, ww


def sfb(V0, V1, N):
    """sfb.m, literal (used by ComputeNow and the tests)"""
    N0, N1 = len(V0), len(V1)
    S, P, T = (N - N0) // 2, (N - N1) // 2, (N0 + N1 - N) // 2 - 1
    tr = _trans(T)
    Y0 = np.zeros(N, complex)
    Y0[0] = V0[0]
    k = np.arange(1, P + 1)
    Y0[k] = V0[k]
    kt = np.arange(1, T + 1)
    Y0[P + kt] = V0[P + kt] * tr
    Y0[N - P - kt] = V0[N0 - P - kt] * tr
    Y0[N - k] = V0[N0 - k]
    Y1 = np.zeros(N, complex)
    Y1[P + kt] = V1[kt] * tr[T - kt]
    ks = np.arange(1, S + 1)
    Y1[P + T + ks] = V1[T + ks]
    if N % 2 == 0:
        Y1[N // 2] = V1[N1 // 2]
    Y1[N - P - T - ks] = V1[N1 - T - ks]
    Y1[N - P - kt] = V1[N1 - kt] * tr[T - kt]
    return Y0 + Y1


def nextpow2(k):
    return int(2 ** math.ceil(math.log2(k)))


def max_levels(Q, r, N):
    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    return int(math.floor(math.log(beta * N / 8) / math.log(1 / alpha)))


def compute_now(N, Q, r, J):
    """ComputeNow.m (non radix-2): l2 norm of the wavelets of each subband"""
    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    lens = [2 * round_half(beta * alpha ** (j - 1) * N / 2) for j in range(1, J + 1)]
    lens.append(2 * round_half(alpha ** J * N / 2))
    now = np.zeros(J + 1)
    for i in range(J + 1):
        w = [np.zeros(m, complex) for m in lens]
        w[i][:] = 1 / math.sqrt(lens[i])
        Y = w[J]
        for j in range(J, 0, -1):
            M = 2 * round_half(alpha ** (j - 1) * N / 2)
            Y = sfb(Y, w[j - 1], M)
        now[i] = math.sqrt(np.sum(np.abs(Y) ** 2))
    return now


def round_half(x):
    return int(math.floor(x + 0.5))


class TQWTRadix2:
    """tqwt_radix2 / itqwt_radix2 for signals of length L (padded to N = nextpow2(L)).

    Coefficients are held per group of equal-length subbands: a list of arrays
    (n_epochs, n_subbands_in_group, length); `to_list` / `from_list` convert to the
    MATLAB cell layout w{1..J+1}."""

    def __init__(self, Q, r, J, L):
        self.N = N = nextpow2(L)
        beta = 2 / (Q + 1)
        alpha = 1 - beta / r
        if J > max_levels(Q, r, N):
            raise ValueError("too many levels")
        idx, w = np.arange(N), np.ones(N)
        maps = []
        for j in range(1, J + 1):
            N0 = 2 * round_half(alpha ** j * N / 2)
            N1 = 2 * round_half(beta * alpha ** (j - 1) * N / 2)
            (idx, w), (i1, w1) = afb_map(idx, w, N0, N1)
            maps.append(lps_map(i1, w1, nextpow2(N1)))
        maps.append(lps_map(idx, w, nextpow2(N0)))
        self.lens = [len(m[0]) for m in maps]
        self.groups = []                       # (subband ids, length, idx (g, n), w (g, n))
        for n in sorted(set(self.lens)):
            ids = [j for j, m in enumerate(self.lens) if m == n]
            self.groups.append((ids, n, np.stack([maps[j][0] for j in ids]),
                                np.stack([maps[j][1] for j in ids])))
        # Real signals: work on the non-negative half spectrum (rfft / irfft). Every non-negative
        # subband bin gathers from a non-negative bin of X (checked), so the half maps are exact.
        self.half = []
        for ids, n, i, w in self.groups:
            ih, wh = i[:, :n // 2 + 1], w[:, :n // 2 + 1].copy()
            assert np.all(ih[wh != 0] <= N // 2)
            self.half.append((ih, wh))
        # synthesis = adjoint of the gather, as a sparse (N/2+1) x sum(n/2+1) matrix
        off, rows = 0, []
        for ih, wh in self.half:
            rows.append(np.arange(ih.size) + off)
            off += ih.size
        cols = np.concatenate([ih.ravel() for ih, wh in self.half])
        vals = np.concatenate([wh.ravel() for ih, wh in self.half])
        syn = sparse.csr_matrix((vals, (cols, np.concatenate(rows))), shape=(N // 2 + 1, off))
        syn.eliminate_zeros()
        # dense padded form: output bin k = sum_m Ccat[:, pos[k, m]] * wt[k, m]
        nnz = np.diff(syn.indptr)
        M = int(nnz.max())
        self.pos = np.zeros((N // 2 + 1, M), int)
        self.wt = np.zeros((N // 2 + 1, M))
        for k in range(N // 2 + 1):
            a, b = syn.indptr[k], syn.indptr[k + 1]
            self.pos[k, :b - a] = syn.indices[a:b]
            self.wt[k, :b - a] = syn.data[a:b]
        self.inv_sqrt = np.concatenate([np.full(ih.size, 1 / math.sqrt(g[1]))
                                        for (ih, wh), g in zip(self.half, self.groups)])
        self.wt *= self.inv_sqrt[self.pos]

    def forward(self, x):
        """x: (n_epochs, N) real -> grouped real subbands (uDFTinv of each subband DFT)"""
        X = sfft.rfft(x, self.N, axis=-1) / math.sqrt(self.N)
        return [sfft.irfft(X[:, ih] * wh, n, axis=-1) * math.sqrt(n)
                for (ih, wh), (ids, n, *_) in zip(self.half, self.groups)]

    def inverse(self, coefs):
        """adjoint of forward == itqwt_radix2 (tight frame); returns (n_epochs, N) real"""
        C = np.concatenate([sfft.rfft(c, axis=-1).reshape(c.shape[0], -1) for c in coefs], axis=1)
        Y = (C[:, self.pos] * self.wt).sum(-1)
        return sfft.irfft(Y, self.N, axis=-1) * math.sqrt(self.N)

    def group_values(self, per_subband):
        """per-subband scalars (thresholds) -> list of (g, 1) arrays matching the groups"""
        v = np.asarray(per_subband, float)
        return [v[ids][:, None] for ids, *_ in self.groups]

    def to_list(self, coefs):
        out = [None] * len(self.lens)
        for c, (ids, *_) in zip(coefs, self.groups):
            for k, j in enumerate(ids):
                out[j] = c[:, k]
        return out

    def from_list(self, lst):
        return [np.stack([lst[j] for j in ids], axis=1) for ids, *_ in self.groups]


def soft(x, T):
    """soft.m (real branch)"""
    return np.sign(x) * np.maximum(np.abs(x) - T, 0)


def dual_q(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, nit):
    """dualQ.m (SALSA), vectorised over epochs. x: (n_epochs, L) real"""
    L = x.shape[1]
    t1, t2 = TQWTRadix2(Q1, r1, J1, L), TQWTRadix2(Q2, r2, J2, L)
    N = t1.N
    xp = np.zeros((x.shape[0], N))
    xp[:, :L] = x
    w1, w2 = t1.forward(xp), t2.forward(xp)
    d1 = [np.zeros_like(a) for a in w1]
    d2 = [np.zeros_like(a) for a in w2]
    T1, T2 = t1.group_values(lam1 / (2 * mu)), t2.group_values(lam2 / (2 * mu))
    for _ in range(nit):
        u1 = [soft(w + d, T) - d for w, d, T in zip(w1, d1, T1)]
        u2 = [soft(w + d, T) - d for w, d, T in zip(w2, d2, T2)]
        c = 0.5 * (xp - t1.inverse(u1) - t2.inverse(u2))
        d1, d2 = t1.forward(c), t2.forward(c)
        w1 = [d + u for d, u in zip(d1, u1)]
        w2 = [d + u for d, u in zip(d2, u2)]
    return t1.inverse(w1)[:, :L], t2.inverse(w2)[:, :L]


# ------------------------------------------------------------ Spinky
DEF = dict(Q1=5.5, r1=3, Q2=1.0, r2=3, nit=500, mu=0.005, theta=0.5,
           pre_band=(0.2, 45), pre_order=105, kc_band=(0.8, 4), kc_order=100,
           min_distance_s=1.5, ratio=2.5, lookback_s=1.0, epoch_s=30)


def signal_decomposition(epochs, fs, d=DEF):
    """signal_decomposition.m for a stack of epochs (n_epochs, L) -> transient, oscillatory"""
    L = epochs.shape[1]
    b = firwin(d["pre_order"] + 1, d["pre_band"], pass_zero=False, fs=fs)
    x = filtfilt(b, 1, epochs, axis=-1, padlen=3 * (len(b) - 1))
    J1, J2 = max_levels(d["Q1"], d["r1"], L), max_levels(d["Q2"], d["r2"], L)
    now1, now2 = compute_now(L, d["Q1"], d["r1"], J1), compute_now(L, d["Q2"], d["r2"], J2)
    lam1, lam2 = d["theta"] * now1, (1 - d["theta"]) * now2
    oscil, transit = dual_q(x, d["Q1"], d["r1"], J1, d["Q2"], d["r2"], J2, lam1, lam2, d["mu"], d["nit"])
    return transit, oscil


def kc_detection(transit, threshold, fs, d=DEF):
    """kc_detection.m on one epoch's transient component.
    returns trough sample indices (0-based, on the causally filtered signal), and that signal"""
    x = transit - transit.mean()
    b = firwin(d["kc_order"] + 1, d["kc_band"], pass_zero=False, fs=fs)
    k = lfilter(b, 1, x)
    inv = 1.01 * k.max() - k
    locs, _ = find_peaks(inv, distance=d["min_distance_s"] * fs)
    locs = locs[k[locs] <= threshold]
    lb = int(round(d["lookback_s"] * fs))
    locs = locs[locs + 1 > fs]                      # MATLAB indk > fs (1-based)
    keep = [i for i in locs if k[i] < d["ratio"] * k[i - lb]]
    return np.array(keep, int), k


def kc_extent(k, i):
    """label-free event extent (not part of Spinky, which reports a position only):
    from the zero crossing before the trough to the end of the following positive half-wave"""
    s = i
    while s > 0 and k[s - 1] < 0:
        s -= 1
    e = i
    while e < len(k) - 1 and k[e + 1] < 0:
        e += 1
    while e < len(k) - 1 and k[e + 1] >= 0:
        e += 1
    return s, e
