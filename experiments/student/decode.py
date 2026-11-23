"""
Decoders: per-sample P(spindle) at 100 Hz -> events (plan section 5; docs/student_architecture.md, section 6).

    d0(p, tau)                       threshold -> runs -> SUMOv2's merge_and_filter_spindles (MODA rules)
    d1(p, step, gamma, mode, lam)    change-point SEBBs (Ebbers et al. 2024), re-implemented from the
                                     paper's description (the reference package is AGPL-3.0)
    d2(p)                            two-state semi-Markov Viterbi: spindle 0.3-2.5 s, background >= 0.1 s
    snap(events, b)                  move each onset / offset to the nearest local maximum of b within +-0.2 s

All return an array [n, 3]: onset (s), offset (s), score = mean p inside the event.
"""
import numpy as np

FS = 100
MIN_DUR, MAX_DUR = 0.3, 2.5     # MODA / SUMOv2 keep rule (s)
MIN_GAP = 0.1                   # background dwell (s) for D2


def _sumo():
    import data
    return data.sumo()


def _runs(y):
    d = np.diff(np.r_[0, y.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _score(p, ev):
    """append mean p over each event (columns onset s, offset s)"""
    if not len(ev):
        return np.zeros((0, 3))
    if _slice_means is not None and isinstance(p, np.ndarray) and p.dtype == np.float64 and p.flags.c_contiguous:
        s = np.rint(ev[:, 0] * FS).astype(np.int64)            # rint and round() both round half to even
        e = np.maximum(np.rint(ev[:, 1] * FS).astype(np.int64), s + 1)
        return np.c_[ev[:, :2], _slice_means(p, s, e)]
    sc = [float(p[int(round(a * FS)):max(int(round(b * FS)), int(round(a * FS)) + 1)].mean()) for a, b in ev[:, :2]]
    return np.c_[ev[:, :2], sc]


def _pairwise_py(a, lo, n):
    """numpy's pairwise summation of a[lo:lo + n] (float64), in its order, so the sum is bit-identical"""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    if n <= 128:
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        i = 8
        while i < n - (n % 8):
            r0 += a[lo + i]
            r1 += a[lo + i + 1]
            r2 += a[lo + i + 2]
            r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]
            r5 += a[lo + i + 5]
            r6 += a[lo + i + 6]
            r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise(a, lo, n2) + _pairwise(a, lo + n2, n - n2)


def _slice_means_py(p, s, e):
    """mean of p[s[k]:e[k]] for every k, as numpy computes it (slice clamped to p; empty -> nan)"""
    out = np.empty(len(s))
    for k in range(len(s)):
        lo, hi = min(max(s[k], 0), len(p)), min(max(e[k], 0), len(p))
        n = hi - lo
        out[k] = (0.0 + _pairwise(p, lo, n)) / n if n > 0 else np.nan
    return out


try:
    from numba import njit
    _pairwise = njit(cache=True)(_pairwise_py)
    _slice_means = njit(cache=True)(_slice_means_py)
except ImportError:                                  # the list form in _score is used instead
    _pairwise, _slice_means = _pairwise_py, None


def d0(p, tau=0.5):
    """SUMOv2 decoding with threshold tau (tau = 0.5 is SUMOv2's argmax)"""
    s, e = _runs(np.asarray(p) > tau)
    if not len(s):
        return np.zeros((0, 3))
    ev = _sumo().merge_and_filter_spindles(np.c_[s, e].astype(float) / FS)
    return _score(p, ev)


# ------------------------------------------------------------------ D1: change-point SEBBs

def step_delta(p, step):
    """delta(t) = mean p over [t, t + step/2) - mean p over [t - step/2, t), p piecewise constant;
    step in seconds, half-window rounded to whole samples; edges use the available samples"""
    h = max(int(round(step * FS / 2)), 1)
    c = np.r_[0.0, np.cumsum(p, dtype=float)]
    n = len(p)
    t = np.arange(n + 1)
    hi, lo = np.minimum(t + h, n), np.maximum(t - h, 0)
    after = (c[hi] - c[t]) / np.maximum(hi - t, 1)
    before = (c[t] - c[lo]) / np.maximum(t - lo, 1)
    return after - before            # defined on the n + 1 sample boundaries


def _extrema(d):
    """local maxima and minima of d; flat stretches are collapsed first, so maxima and minima alternate"""
    keep = np.r_[True, np.diff(d) != 0]
    idx = np.flatnonzero(keep)
    v = d[idx]
    sd = np.sign(np.diff(v))
    turn = np.diff(sd)
    mx = idx[np.flatnonzero(turn < 0) + 1]
    mn = idx[np.flatnonzero(turn > 0) + 1]
    return mx, mn


def sebb_boxes(p, step=0.48, gamma=2.0, mode="rel"):
    """change-point SEBBs: onsets = local maxima of delta, offsets = local minima; each onset with the
    next offset is a tentative event, each offset with the next onset a tentative gap. A gap is removed
    (its two events merged) when BOTH neighbouring events' maximum p is within gamma of the gap's
    minimum p:  mode "abs": max_event - min_gap < gamma;  mode "rel": max_event < gamma * min_gap.
    [ours: the paper leaves "both / either" and the sign of the difference open.]
    -> array [n, 3] onset s, offset s, box confidence = mean p in the box"""
    p = np.asarray(p, float)
    d = step_delta(p, step)
    on, off = _extrema(d)
    if not len(on) or not len(off):
        return np.zeros((0, 3))
    # alternate onset -> next offset -> next onset ...; the "next" indices are looked up once
    nxt_off = np.searchsorted(off, on, side="right").tolist()
    nxt_on = np.searchsorted(on, off, side="right").tolist()
    onl, offl = on.tolist(), off.tolist()
    ev, i = [], 0
    while i < len(onl):
        k = nxt_off[i]
        if k >= len(offl):
            break
        ev.append((onl[i], offl[k]))
        i = nxt_on[k]
    if not ev:
        return np.zeros((0, 3))
    # events and the gaps between them tile [on_0, off_last): their max / min p in one pass each.
    # Gaps are never empty (the next onset is strictly after the offset); an event can end at n.
    bounds = np.array(ev, np.int64).ravel()
    pp = np.r_[p, p[-1]]                  # index n only ever starts the unused tail after the last event
    mx = np.maximum.reduceat(pp, bounds).tolist()
    mn = np.minimum.reduceat(pp, bounds).tolist()
    emax, gmax, gmin = mx[0::2], mx[1::2], mn[1::2]
    merged = [[ev[0][0], ev[0][1]]]
    cur = emax[0]                         # max p over the current merged box (events and inner gaps)
    for j in range(1, len(ev)):
        a, b = ev[j]
        gap_min, left, right = gmin[j - 1], cur, emax[j]
        if mode == "abs":
            ok = (left - gap_min < gamma) and (right - gap_min < gamma)
        else:
            ok = (left < gamma * gap_min) and (right < gamma * gap_min)
        if ok:
            merged[-1][1] = b
            cur = max(cur, gmax[j - 1], right)
        else:
            merged.append([a, b])
            cur = right
    box = np.array(merged, float) / FS
    return _score(p, box)


def d1(p, step=0.48, gamma=2.0, mode="rel", lam=0.5):
    """SEBB boxes with confidence > lam, then the MODA keep rule 0.3-2.5 s (no merging)"""
    b = sebb_boxes(p, step, gamma, mode)
    if not len(b):
        return b
    dur = b[:, 1] - b[:, 0]
    return b[(b[:, 2] > lam) & (dur >= MIN_DUR) & (dur <= MAX_DUR)]


# ------------------------------------------------------------------ D2: semi-Markov Viterbi

def _d2_core(p, min_s=MIN_DUR, max_s=MAX_DUR, min_b=MIN_GAP, eps=1e-6):
    """MAP segmentation of [0, n) into alternating background / spindle segments, spindle durations in
    [min_s, max_s], background >= min_b (the first and the last segment may be background of any length).
    Emissions log p (spindle) and log(1 - p) (background); uniform duration prior. With C_S, C_B the
    cumulative emission sums and segment boundaries b = 0..n:
        V_S(b) = C_S(b) + max_{a in [b - Dmax, b - Dmin]} [V_B(a) - C_S(a)],     V_B(0) = 0
        V_B(b) = C_B(b) + max(0, max_{1 <= a <= b - Bmin} [V_S(a) - C_B(a)])
    Exact. Every dependency reaches back >= Bmin samples, so blocks of Bmin boundaries are computed at
    once. -> events [n, 3] (onset s, offset s, mean p)"""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    n = len(p)
    Dmin, Dmax, Bmin = int(round(min_s * FS)), int(round(max_s * FS)), int(round(min_b * FS))
    CS = np.r_[0.0, np.cumsum(np.log(p))]
    CB = np.r_[0.0, np.cumsum(np.log1p(-p))]
    if _d2_forward is not None:
        VS, aS, aB, PMv, PMi = _d2_forward(CS, CB, n, Dmin, Dmax, Bmin)
        return _d2_back(n, VS, CB, aS, aB, PMv, PMi)
    NEG = -np.inf
    VS, VB = np.full(n + 1, NEG), np.full(n + 1, NEG)
    aS, aB = np.zeros(n + 1, np.int64), np.zeros(n + 1, np.int64)
    VB[0] = 0.0
    G = np.full(n + 1, NEG)                    # V_B(a) - C_S(a)
    G[0] = 0.0
    PMv = np.full(n + 1, NEG)                  # prefix max over a in [1, i] of V_S(a) - C_B(a)
    PMi = np.zeros(n + 1, np.int64)
    W = Dmax - Dmin + 1
    for t0 in range(1, n + 1, Bmin):
        b = np.arange(t0, min(t0 + Bmin, n + 1))
        # background ending at b
        j = b - Bmin
        prev = np.where(j >= 1, PMv[np.maximum(j, 0)], NEG)
        use = prev > 0.0
        VB[b] = CB[b] + np.where(use, prev, 0.0)
        aB[b] = np.where(use, PMi[np.maximum(j, 0)], 0)
        # spindle ending at b: window max of G over a in [b - Dmax, b - Dmin]
        hi = b - Dmin
        ok = hi >= 0
        if ok.any():
            bb = b[ok]
            lo = bb - Dmax
            start = lo[0]
            seg = G[max(start, 0):hi[ok][-1] + 1]
            if start < 0:
                seg = np.r_[np.full(-start, NEG), seg]
            win = np.lib.stride_tricks.sliding_window_view(seg, W)[:len(bb)]
            k = win.argmax(1)
            best = win[np.arange(len(bb)), k]
            VS[bb] = CS[bb] + best
            aS[bb] = lo + k
        G[b] = VB[b] - CS[b]
        h = VS[b] - CB[b]                      # prefix max of h, carried over from the last block
        with np.errstate(invalid="ignore"):
            cm = np.maximum.accumulate(h)
        first = np.r_[0, np.flatnonzero(np.diff(cm) > 0) + 1]
        ci = b[first][np.searchsorted(first, np.arange(len(b)), side="right") - 1]
        carry = PMv[t0 - 1] >= cm
        PMv[b] = np.where(carry, PMv[t0 - 1], cm)
        PMi[b] = np.where(carry, PMi[t0 - 1], ci)
    return _d2_back(n, VS, CB, aS, aB, PMv, PMi)


def _d2_forward_py(CS, CB, n, Dmin, Dmax, Bmin):
    """the recursion of _d2_core one boundary at a time (compiled with numba when available): the same
    additions and comparisons as the block form, first maximum on ties, so the result is identical"""
    NEG = -np.inf
    VS = np.full(n + 1, NEG)
    aS, aB = np.zeros(n + 1, np.int64), np.zeros(n + 1, np.int64)
    G = np.full(n + 1, NEG)
    G[0] = 0.0
    PMv = np.full(n + 1, NEG)
    PMi = np.zeros(n + 1, np.int64)
    mv, mi = NEG, 0
    for b in range(1, n + 1):
        j = b - Bmin
        prev = PMv[j] if j >= 1 else NEG
        if prev > 0.0:
            vb = CB[b] + prev
            aB[b] = PMi[j]
        else:
            vb = CB[b] + 0.0
        hi = b - Dmin
        if hi >= 0:
            lo = b - Dmax
            best, k = NEG, 0
            for a in range(lo, hi + 1):
                if a >= 0 and G[a] > best:
                    best, k = G[a], a - lo
            VS[b] = CS[b] + best
            aS[b] = lo + k
        G[b] = vb - CS[b]
        h = VS[b] - CB[b]
        if h > mv:
            mv, mi = h, b
        PMv[b], PMi[b] = mv, mi
    return VS, aS, aB, PMv, PMi


try:
    from numba import njit
    _d2_forward = njit(cache=True)(_d2_forward_py)
except ImportError:                              # the block form in _d2_core is used instead
    _d2_forward = None


def _d2_back(n, VS, CB, aS, aB, PMv, PMi):
    """backtrack the D2 optimum -> segments [n, 2] in samples"""
    # the last segment: a spindle ending at n, or background of any length after a spindle / from 0
    end_b = CB[n] + max(0.0, PMv[n])
    state, bpos = ("S", n) if VS[n] > end_b else ("Bend", n)
    ev = []
    while bpos > 0:
        if state == "S":
            a = int(aS[bpos])
            ev.append((a, bpos))
            state, bpos = "B", a
        elif state == "Bend":
            a = int(PMi[n]) if PMv[n] > 0.0 else 0
            state, bpos = "S", a
        else:
            a = int(aB[bpos])
            state, bpos = "S", a
    return np.array(ev[::-1], np.int64).reshape(-1, 2)


def d2(p, min_s=MIN_DUR, max_s=MAX_DUR, min_b=MIN_GAP):
    """exact D2 on the whole recording, solved only where a spindle segment can be: every spindle segment
    of the optimum contains a sample with p >= 0.5 (otherwise relabelling it background scores higher and
    keeps every duration legal), so it lies within max_s of such a sample. Outside those windows (padded by
    max_s + min_b) the optimum is background; each window is solved by _d2_core with free background ends."""
    p = np.asarray(p, float)
    n = len(p)
    hot = np.flatnonzero(p >= 0.5)
    if not len(hot):
        return np.zeros((0, 3))
    pad = int(round((max_s + min_b) * FS))
    starts = np.maximum(hot - pad, 0)
    ends = np.minimum(hot + pad + 1, n)
    brk = np.flatnonzero(starts[1:] > ends[:-1]) + 1
    ev = []
    for a, b in zip(starts[np.r_[0, brk]], ends[np.r_[brk - 1, len(hot) - 1]]):
        ev.append(_d2_core(p[a:b], min_s, max_s, min_b) + a)
    ev = np.concatenate(ev).astype(float) / FS
    return _score(p, ev)


def snap(events, b, radius=0.2):
    """move each onset and offset to the highest local maximum of b within +-radius s (unchanged if none);
    an event that would become shorter than MIN_DUR keeps its original boundaries"""
    if not len(events):
        return events
    b = np.asarray(b, float)
    r = int(round(radius * FS))
    out = events.copy()
    for k, (on, off) in enumerate(events[:, :2]):
        new = []
        for t in (on, off):
            i = int(round(t * FS))
            lo, hi = max(i - r, 1), min(i + r + 1, len(b) - 1)
            seg = b[lo:hi]
            pk = np.flatnonzero((seg >= b[lo - 1:hi - 1]) & (seg >= b[lo + 1:hi + 1]))
            new.append((lo + pk[np.argmax(seg[pk])]) / FS if len(pk) else t)
        if new[1] - new[0] >= MIN_DUR:
            out[k, :2] = new
    return out
