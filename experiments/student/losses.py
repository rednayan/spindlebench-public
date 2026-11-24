"""
Training losses (plan 4.4, 4.5; equations in docs/student_architecture.md, section 5).

All losses take a sample mask m [N, T] (valid = N2, clean, and inside the central 30 s of the crop)
and average over the masked samples of the batch.

    noisy_annotator   sum_r CE(y_r, A_r^T p) + lam * mean_r tr A_r       (Tanno et al. 2019, eq. 4 and Listing 1:
                      CE summed over annotators, trace averaged over them, lam = 0.01; amendment 8)
    soft_ce           CE(s, p) with a soft or hard target s in [0, 1]    (student_soft / _mv / _ds / per-teacher)
    gdl               SUMOv2's generalised Dice loss, masked             (ablation)
    boundary_bce      BCE of head 2 against +-W samples around consensus boundaries
                      (Mitsuoka & Hotta 2026, Eq. 1)
    segment_cdf       squared L2 distance between the CDF of the normalised spindle probability inside
                      each consensus event (margin delta) and the uniform CDF (Mitsuoka & Hotta 2026, Eq. 2-3)
"""
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-7
BOUNDARY_HALF = 10      # samples = +-100 ms  [ours: the source used +-5 video frames]
SEGMENT_MARGIN = 10     # delta, samples = 100 ms  [ours: the source used delta = 5 frames]


def _masked_mean(v, m):
    return (v * m).sum() / m.sum().clamp_min(1.0)


def teacher_probs(p2, A):
    """q[n, r, j, t] = sum_i p2[n, i, t] A[n, r, i, j, t]: the probability that teacher r says j"""
    return torch.einsum("nit,nrijt->nrjt", p2, A)


def noisy_annotator(logits, A, Y, m, lam):
    """logits [N, 2, T]; A [N, R, 2, 2, T]; Y [N, R, T] in {0, 1}; m [N, T] -> (loss, parts)"""
    p2 = torch.softmax(logits, 1)
    q = teacher_probs(p2, A).clamp(EPS, 1.0)
    y = Y.long()
    logq = torch.log(torch.where(y == 1, q[:, :, 1], q[:, :, 0]))   # [N, R, T]
    ce = _masked_mean(-logq.sum(1), m)
    d0, d1 = A[:, :, 0, 0].mean(1), A[:, :, 1, 1].mean(1)        # mean over teachers, [N, T]
    tr = _masked_mean(d0 + d1, m)                                # trace of the mean CM = mean_r tr A_r
    # diagnostics (amendment 8): the mean CM's diagonal weighted by the student's own class probability.
    # A(t) is per sample; row i only acts where class i is present (Zhang et al. 2020, Theorem 1 is per
    # sample and about the true class), so an unweighted mean of A[1,1] over mostly spindle-free samples
    # says nothing about the teachers' recall.
    w = p2.detach() * m[:, None]
    diag0 = (w[:, 0] * d0.detach()).sum() / w[:, 0].sum().clamp_min(EPS)
    diag1 = (w[:, 1] * d1.detach()).sum() / w[:, 1].sum().clamp_min(EPS)
    return ce + lam * tr, dict(ce=ce.detach(), trace=tr.detach(), diag0=diag0, diag1=diag1)


def soft_ce(logits, s, m):
    """-[s log p1 + (1 - s) log p0], s [N, T] in [0, 1]"""
    logp = torch.log_softmax(logits, 1)
    return _masked_mean(-(s * logp[:, 1] + (1 - s) * logp[:, 0]), m)


def gdl(logits, y, m, smooth=1e-5):
    """SUMOv2's GeneralizedDiceLoss (Sudre et al. 2017; class weights 1 / volume^2, weight 1 for an
    empty class), restricted to masked samples, mean over the batch"""
    p = torch.softmax(logits, 1) * m[:, None]
    oh = torch.stack([1 - y, y], 1).float() * m[:, None]
    inter, union = (p * oh).sum(2), (p + oh).sum(2)
    w = 1.0 / oh.sum(2) ** 2
    w[torch.isinf(w)] = 1.0
    dice = (2 * (w * inter).sum(1) + smooth) / ((w * union).sum(1) + smooth)
    return (1 - dice).mean()


def runs(y):
    """(start, stop) of runs of 1 in a 1-D numpy array"""
    d = np.diff(np.r_[0, y.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def consensus_events(mv, valid, min_len=30, max_len=250):
    """per crop: consensus events (start, stop) that lie wholly inside valid samples and do not touch
    the crop edges (an event cut by the crop has a false boundary), kept 0.3-2.5 s as MODA"""
    out = []
    for y, v in zip(mv, valid):
        s, e = runs(y)
        keep = [(a, b) for a, b in zip(s, e)
                if a > 0 and b < len(y) and min_len <= b - a <= max_len and v[a:b].all()]
        out.append(keep)
    return out


def boundary_targets(events, n, T, half=BOUNDARY_HALF):
    b = np.zeros((n, T), np.float32)
    for i, ev in enumerate(events):
        for a, e in ev:
            for c in (a, e):
                b[i, max(c - half, 0):c + half + 1] = 1.0
    return b


def boundary_bce(b_logit, b_target, m):
    return _masked_mean(F.binary_cross_entropy_with_logits(b_logit, b_target, reduction="none"), m)


def segment_cdf(p, events, margin=SEGMENT_MARGIN):
    """mean over events of (1/L) sum_j (CDF(p_hat)[j] - CDF(uniform)[j])^2, p_hat = the spindle
    probability on [s + margin, e - margin) normalised to sum 1; events with L <= 0 are skipped"""
    terms = []
    for i, ev in enumerate(events):
        for a, e in ev:
            L = e - a - 2 * margin
            if L <= 0:
                continue
            seg = p[i, a + margin:e - margin]
            ph = seg / seg.sum().clamp_min(EPS)
            cdf = torch.cumsum(ph, 0)
            ref = torch.arange(1, L + 1, dtype=p.dtype, device=p.device) / L
            terms.append(((cdf - ref) ** 2).mean())
    if not terms:
        return p.sum() * 0.0
    return torch.stack(terms).mean()


@torch.no_grad()
def pu_keep_mask(x_orig, Y, fs=100, sigma_mult=2.0, smooth_len=41):
    """Narrow PU keep-mask [N, T] bool for N2 crops (computed on the UNMIXED signal x_orig):
      - n_votes >= 2: keep = True (trusted positives)
      - n_votes == 1: keep = False (ambiguous single-teacher vote)
      - n_votes == 0 and sustained sigma-dominant burst (sigma > sigma_mult * crop_median,
        sigma > alpha, sigma > beta over a ~0.4 s window): keep = False (plausible teacher miss)
      - all other n_votes == 0 (quiet N2, short blips, alpha/beta bursts): keep = True (hard negatives)"""
    n_votes = Y.sum(dim=1)
    T = x_orig.shape[-1]
    X = torch.fft.rfft(x_orig.float(), dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=x_orig.device)

    def band_pow(lo, hi):
        fmask = (freqs >= lo) & (freqs <= hi)
        xb = torch.fft.irfft(X * fmask, n=T, dim=-1)
        return F.avg_pool1d((xb ** 2).unsqueeze(1), smooth_len, stride=1, padding=smooth_len // 2)[:, 0]

    sig_p = band_pow(11.0, 16.0)
    alp_p = band_pow(8.0, 10.5)
    bet_p = band_pow(18.0, 30.0)
    med = sig_p.sort(dim=-1).values[:, T // 2 : T // 2 + 1].clamp_min(1e-8)
    plausible_miss = (n_votes == 0) & (sig_p > sigma_mult * med) & (sig_p > alp_p) & (sig_p > bet_p)
    return ~((n_votes == 1) | plausible_miss)
