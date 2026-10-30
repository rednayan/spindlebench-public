"""
Self-checks of the TQWT used by the Spinky port (no MATLAB needed):
  1. gather-map forward == literal tqwt_radix2 cascade (afb.m / lps.m transcribed on values)
  2. literal itqwt_radix2 (sfb.m cascade) == port's inverse  ->  perfect reconstruction
  3. ComputeNow-style wavelet norms of the radix-2 transform are ~1 for interior subbands
    .venv/Scripts/python detectors/kcomplex_rules/check_spinky.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spinky as sk  # noqa: E402


def afb(X, N0, N1):
    N = len(X)
    P, T, S = (N - N1) // 2, (N0 + N1 - N) // 2 - 1, (N - N0) // 2
    tr = sk._trans(T)
    V0 = np.zeros(N0, complex)
    V0[0] = X[0]
    V0[1:P + 1] = X[1:P + 1]
    k = np.arange(1, T + 1)
    V0[P + k] = X[P + k] * tr
    V0[N0 // 2] = 0
    V0[N0 - P - k] = X[N - P - k] * tr
    kp = np.arange(1, P + 1)
    V0[N0 - kp] = X[N - kp]
    V1 = np.zeros(N1, complex)
    V1[k] = X[P + k] * tr[::-1]
    ks = np.arange(1, S + 1)
    V1[T + ks] = X[P + T + ks]
    if N % 2 == 0:
        V1[N1 // 2] = X[N // 2]
    V1[N1 - T - ks] = X[N - P - T - ks]
    V1[N1 - k] = X[N - P - k] * tr[::-1]
    return V0, V1


def lps(X, N0):
    i, w = sk.lps_map(np.arange(len(X)), np.ones(len(X)), N0)
    return X[i] * w


def tqwt_radix2(x, Q, r, J):
    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    N = sk.nextpow2(len(x))
    X = np.fft.fft(x, N) / math.sqrt(N)
    w = []
    for j in range(1, J + 1):
        N0 = 2 * sk.round_half(alpha ** j * N / 2)
        N1 = 2 * sk.round_half(beta * alpha ** (j - 1) * N / 2)
        X, W = afb(X, N0, N1)
        W = lps(W, sk.nextpow2(N1))
        w.append(np.fft.ifft(W) * math.sqrt(len(W)))
    X = lps(X, sk.nextpow2(N0))
    w.append(np.fft.ifft(X) * math.sqrt(len(X)))
    return w


def itqwt_radix2(w, Q, r, L):
    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    J = len(w) - 1
    N = sk.nextpow2(L)
    Y = np.fft.fft(w[J]) / math.sqrt(len(w[J]))
    Y = lps(Y, 2 * sk.round_half(alpha ** J * N / 2))
    for j in range(J, 0, -1):
        W = np.fft.fft(w[j - 1]) / math.sqrt(len(w[j - 1]))
        W = lps(W, 2 * sk.round_half(beta * alpha ** (j - 1) * N / 2))
        Y = sk.sfb(Y, W, 2 * sk.round_half(alpha ** (j - 1) * N / 2))
    return (np.fft.ifft(Y) * math.sqrt(N))[:L]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    L = 6000
    x = rng.standard_normal(L) * 30
    for Q, r in ((5.5, 3), (1.0, 3)):
        J = sk.max_levels(Q, r, L)
        t = sk.TQWTRadix2(Q, r, J, L)
        xp = np.zeros((1, t.N))
        xp[0, :L] = x
        lit = tqwt_radix2(xp[0], Q, r, J)
        fast = t.to_list(t.forward(xp))
        e_fwd = max(np.max(np.abs(a - b[0])) for a, b in zip(lit, fast))
        imag = max(np.max(np.abs(a.imag)) for a in lit)
        rec_lit = itqwt_radix2([b[0] for b in fast], Q, r, t.N).real
        rec_fast = t.inverse(t.from_list(fast))[0]
        print(f"Q={Q} J={J}: |fwd-literal|={e_fwd:.2e}  max|imag literal|={imag:.2e}  "
              f"|inv-literal inv|={np.max(np.abs(rec_lit - rec_fast)):.2e}  "
              f"PR error={np.max(np.abs(rec_fast - xp[0])):.2e}")
        now = sk.compute_now(L, Q, r, J)
        print("   ComputeNow first/last:", np.round(now[:3], 3), np.round(now[-2:], 3))
