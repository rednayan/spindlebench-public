"""
The teacher-aware student (plan section 4; full description with equations in
docs/student_architecture.md).

    Student(trunk="S"|"M", stream_b=True, noise="local"|"global"|None, boundary=True, ...)

  forward(x [N, T]) -> dict
      logits    [N, 2, T]   head 1, background / spindle (trunk S: before SUMOv2's moving average)
      p         [N, T]      P(spindle | x) = softmax(logits)[:, 1]
      b_logit   [N, T]      head 2, boundary channel (if boundary)
      A         [N, R, 2, 2, T]  teacher confusion matrices, A[..., i, j, t] = P(y_r = j | y = i) (if noise)
  predict(x) -> p [N, T] with SUMOv2's inference post-processing (moving average of 42 logits for trunk S)

Trunk S is SUMOv2's U-Net (Grieger et al. 2026; Kaulen et al. 2022), re-implemented module for module
so that SUMOv2's state_dict loads into it unchanged (checked in tests); it is trained from scratch here.
SUMOv2 code: MIT licence, (c) 2022 Lars Kaulen, Justus T. C. Schwabedal, Stephan Bialonski.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FS = 100
N_TEACHERS = 4


# ------------------------------------------------------------------ trunk S: SUMOv2's U-Net

class SingleConv(nn.Module):
    """conv -> activation -> batch norm (SUMOv2 order)"""

    def __init__(self, c_in, c_out, kernel_size, dilation):
        super().__init__()
        self.single_conv = nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size, padding="same", dilation=dilation),
                                         nn.ReLU(), nn.BatchNorm1d(c_out))

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation):
        super().__init__()
        self.double_conv = nn.Sequential(SingleConv(c_in, c_out, kernel_size, dilation),
                                         SingleConv(c_out, c_out, kernel_size, dilation))

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, c_in, c_out, pool, kernel_size, dilation):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool1d(pool), DoubleConv(c_in, c_out, kernel_size, dilation))

    def forward(self, x):
        return self.maxpool_conv(x)


class Decoder(nn.Module):
    """nearest up-sampling -> SingleConv (kernel = scale factor) -> concat skip -> DoubleConv"""

    def __init__(self, c_in, c_out, scale, kernel_size):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale)
        self.single_conv = SingleConv(c_in, c_out, scale, 1)
        self.double_conv = DoubleConv(c_in, c_out, kernel_size, 1)

    def forward(self, x1, x2):
        x1 = self.single_conv(self.up(x1))
        return self.double_conv(torch.cat([x2, x1], dim=1))


class TrunkS(nn.Module):
    """SUMOv2 final.yaml: depth 2, 16 channels, pools [4, 4], kernel 5, dilation 2 (encoders), ReLU,
    2 classes, logits through ReLU (as SUMOv2's `dense`), moving average 42 at inference only."""
    depth, chs, pools, kernel, dilation, moving_avg = 2, 16, (4, 4), 5, 2, 42

    def __init__(self):
        super().__init__()
        c, k, d = self.chs, self.kernel, self.dilation
        self.inc = DoubleConv(1, c, k, d)
        self.encoders = nn.ModuleList([Encoder(c * 2 ** i, c * 2 ** (i + 1), self.pools[i], k, d)
                                       for i in range(self.depth)])
        self.decoders = nn.ModuleList([Decoder(c * 2 ** (i + 1), c * 2 ** i, self.pools[i], k)
                                       for i in reversed(range(self.depth))])
        self.dense = nn.Sequential(nn.Conv1d(c, 2, kernel_size=1, padding="same"), nn.ReLU())
        self.out_channels = c                      # decoder output, feeds head 2
        self.bottleneck_channels = c * 2 ** self.depth
        self.stride = int(np.prod(self.pools))

    def forward(self, x):
        """x [N, T] -> (logits [N, 2, T], decoder features [N, 16, T], bottleneck [N, 64, T'])"""
        n = x.shape[1]
        x = x.unsqueeze(1)
        ext = int(math.ceil(n / self.stride) * self.stride - n)
        x = F.pad(x, (ext // 2, ext // 2 + ext % 2), mode="reflect")
        skips = []
        x = self.inc(x)
        for enc in self.encoders:
            skips.append(x)
            x = enc(x)
        bottleneck = x
        for dec, s in zip(self.decoders, reversed(skips)):
            x = dec(x, s)
        x = x[:, :, ext // 2: ext // 2 + n]
        return self.dense(x), x, bottleneck, ext // 2

    def postprocess(self, logits):
        """SUMOv2's moving average of the logits (zero padding, window 42, stride 1)"""
        s = self.moving_avg - 1
        return F.avg_pool1d(F.pad(logits, (s // 2, s // 2 + s % 2)), self.moving_avg, stride=1)


# ------------------------------------------------------------------ trunk M

class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in, c_out, k, dilation=1):
        super().__init__(nn.Conv1d(c_in, c_out, k, padding="same", dilation=dilation, bias=False),
                         nn.BatchNorm1d(c_out), nn.ReLU())


class TrunkM(nn.Module):
    """4-level U-Net: stem k9 -> [2 x conv(k7, dilation 2)+BN+ReLU] per encoder level, channels
    32-64-128-256, max-pool 2, nearest up-sampling with skips, [2 x conv(k7)+BN+ReLU] per decoder level,
    full-resolution output; optional BiGRU (128 per direction) at the bottleneck (context ablation).
    Encoder dilation 2 as SUMOv2's (amendment 2026-11-04: with dilation 1 the receptive field is 2.84 s,
    shorter than trunk S's 4.00 s; with 2 it is 4.64 s at the same parameter count)."""
    channels, kernel, stem_kernel, dilation = (32, 64, 128, 256), 7, 9, 2

    def __init__(self, bigru=False):
        super().__init__()
        ch, k = self.channels, self.kernel
        self.stem = ConvBNReLU(1, ch[0], self.stem_kernel)
        d = self.dilation
        self.down = nn.ModuleList([nn.Sequential(ConvBNReLU(ch[0], ch[0], k, d), ConvBNReLU(ch[0], ch[0], k, d))] +
                                  [nn.Sequential(ConvBNReLU(ch[i - 1], ch[i], k, d), ConvBNReLU(ch[i], ch[i], k, d))
                                   for i in range(1, len(ch))])
        self.up = nn.ModuleList([nn.Sequential(ConvBNReLU(ch[i] + ch[i - 1], ch[i - 1], k),
                                               ConvBNReLU(ch[i - 1], ch[i - 1], k))
                                 for i in reversed(range(1, len(ch)))])
        self.bigru = nn.GRU(ch[-1], 128, batch_first=True, bidirectional=True) if bigru else None
        self.gru_proj = nn.Conv1d(256, ch[-1], 1) if bigru else None
        self.dense = nn.Conv1d(ch[0], 2, 1)
        self.out_channels = ch[0]
        self.bottleneck_channels = ch[-1]
        self.stride = 2 ** (len(ch) - 1)

    def forward(self, x):
        """x [N, T] (or [N, C, T] when stream B also feeds the trunk)"""
        n = x.shape[-1]
        x = x.unsqueeze(1) if x.dim() == 2 else x
        ext = int(math.ceil(n / self.stride) * self.stride - n)
        x = F.pad(x, (ext // 2, ext // 2 + ext % 2), mode="reflect")
        x = self.stem(x)
        skips = []
        for i, blk in enumerate(self.down):
            if i:
                x = F.max_pool1d(x, 2)
            x = blk(x)
            skips.append(x)
        x = skips.pop()
        if self.bigru is not None:
            h, _ = self.bigru(x.transpose(1, 2))
            x = x + self.gru_proj(h.transpose(1, 2))
        bottleneck = x
        for blk in self.up:
            s = skips.pop()
            x = blk(torch.cat([F.interpolate(x, size=s.shape[-1], mode="nearest"), s], dim=1))
        x = x[:, :, ext // 2: ext // 2 + n]
        return self.dense(x), x, bottleneck, ext // 2

    def postprocess(self, logits):
        return logits


# ------------------------------------------------------------------ stream B: Gabor -> envelope -> sPCEN

def fft_causal_conv(xp, h):
    """y[t] = sum_k h[c, k] xp[n, c, t + L - 1 - k] for t = 0..T-1, xp [N, C, T + L - 1], h [C, L]"""
    L = h.shape[-1]
    n = xp.shape[-1] + L - 1
    nfft = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(xp, nfft) * torch.fft.rfft(h, nfft)[None], nfft)
    return y[..., L - 1: xp.shape[-1]]


def _logit(v, lo, hi):
    return math.log((v - lo) / (hi - v))


class StreamB(nn.Module):
    """8 learnable complex Gabor filters -> squared modulus -> learnable Gaussian low-pass -> per-channel
    sPCEN (LEAF, Zeghidour et al. 2021). 56 trainable parameters:
        f_k, sigma_k (16)  centre frequency in [4, 24] Hz, Gaussian width in [0.05, 0.4] s (sigmoid)
        lp_k (8)           low-pass width in [0.02, 0.5] s (sigmoid)
        alpha, delta, r, s (4 x 8)   PCEN gain exponent, bias, root, smoothing coefficient
    """
    F_INIT = (11.5, 12.5, 13.5, 15.0, 9.5, 10.5, 7.0, 20.0)
    F_RANGE, SIGMA_RANGE, LP_RANGE = (4.0, 24.0), (0.05, 0.4), (0.02, 0.5)
    GABOR_HALF, LP_HALF, EMA_LEN = 50, 150, 2000      # samples: ±0.5 s, ±1.5 s, 20 s
    EPS = 1e-6

    def __init__(self, pcen=True):
        super().__init__()
        K = len(self.F_INIT)
        self.f_raw = nn.Parameter(torch.tensor([_logit(f, *self.F_RANGE) for f in self.F_INIT]))
        self.sigma_raw = nn.Parameter(torch.full((K,), _logit(0.12, *self.SIGMA_RANGE)))
        self.lp_raw = nn.Parameter(torch.full((K,), _logit(0.10, *self.LP_RANGE)))
        self.alpha = nn.Parameter(torch.full((K,), 0.96))
        self.delta = nn.Parameter(torch.full((K,), 2.0))
        self.r = nn.Parameter(torch.full((K,), 0.5))
        self.s_raw = nn.Parameter(torch.full((K,), _logit(0.004, 0.0, 1.0)))
        self.pcen = pcen
        self.n_channels = K

    @staticmethod
    def _range(raw, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(raw)

    def centre_frequencies(self):
        return self._range(self.f_raw, *self.F_RANGE)

    def gabor(self):
        """complex kernels as (real, imag) [K, 1, 2H+1]; unit gain at the centre frequency"""
        t = torch.arange(-self.GABOR_HALF, self.GABOR_HALF + 1, dtype=self.f_raw.dtype, device=self.f_raw.device) / FS
        f = self.centre_frequencies()[:, None]
        sig = self._range(self.sigma_raw, *self.SIGMA_RANGE)[:, None]
        g = torch.exp(-0.5 * (t / sig) ** 2)
        g = g / g.sum(1, keepdim=True)
        return (g * torch.cos(2 * math.pi * f * t))[:, None], (g * torch.sin(2 * math.pi * f * t))[:, None]

    def lowpass(self):
        t = torch.arange(-self.LP_HALF, self.LP_HALF + 1, dtype=self.f_raw.dtype, device=self.f_raw.device) / FS
        sig = self._range(self.lp_raw, *self.LP_RANGE)[:, None]
        g = torch.exp(-0.5 * (t / sig) ** 2)
        return (g / g.sum(1, keepdim=True))[:, None]

    def envelope(self, x):
        """x [N, T] -> E [N, K, T] = Gaussian-smoothed |x * g_k|^2 (reflect padding for the Gabor filters,
        edge replication for the low-pass); FFT convolutions, equal to F.conv1d up to float rounding"""
        re, im = self.gabor()
        g = torch.cat([re, im], 0)[:, 0]                             # [2K, 2H+1]
        xp = F.pad(x.unsqueeze(1), (self.GABOR_HALF,) * 2, mode="reflect")
        y = fft_causal_conv(xp, g.flip(-1))                          # cross-correlation, as conv1d
        e = y[:, :self.n_channels] ** 2 + y[:, self.n_channels:] ** 2
        ep = F.pad(e, (self.LP_HALF,) * 2, mode="replicate")
        return fft_causal_conv(ep, self.lowpass()[:, 0].flip(-1))

    def smoother(self, e):
        """M(t) = (1 - s) M(t-1) + s E(t) with M(0) = E(0): a causal exponential filter, computed as the
        linear convolution of E (left-padded with E(0)) with its impulse response s (1 - s)^k, truncated at
        EMA_LEN samples and renormalised to unit gain; by FFT (exact up to float rounding)"""
        s = torch.sigmoid(self.s_raw)[:, None]
        k = torch.arange(self.EMA_LEN, dtype=e.dtype, device=e.device)[None]
        h = s * (1 - s) ** k
        h = h / h.sum(1, keepdim=True)                             # [K, L]
        return fft_causal_conv(F.pad(e, (self.EMA_LEN - 1, 0), mode="replicate"), h)

    def forward(self, x):
        e = self.envelope(x)
        if not self.pcen:
            return torch.log(e + self.EPS)
        m = self.smoother(e)
        alpha = self.alpha.clamp(0.0, 1.0)[:, None]
        delta = self.delta.clamp_min(self.EPS)[:, None]
        r = self.r.clamp(self.EPS, 1.0)[:, None]
        return (e / (self.EPS + m) ** alpha + delta) ** r - delta ** r


# ------------------------------------------------------------------ teacher-noise branch

DIAG_INIT = 0.85


class NoiseBranch(nn.Module):
    """per-sample, per-teacher 2x2 confusion matrices A_r(t) (rows: true class, columns: teacher's label).
    Input: stream-B maps and/or the trunk bottleneck up-sampled to full resolution. Three dilated convs
    (k7, 32 ch, dilation 1/2/4, BN, ReLU); a 1x1 conv to R x 2 x 2 logits; softmax over each row.
    The 1x1 conv starts with zero weights and a bias of log(0.85 / 0.15) on the diagonal, so at
    initialisation A_r(t) = [[0.85, 0.15], [0.15, 0.85]] everywhere (MaDL / Annot-Mix diagonal init).

    Anchored variant (anchor = [R, 2, 2] row-stochastic tables, e.g. the Dawid-Skene confusion matrices;
    [ours, rules4-fix pilot]): logits = log(anchor) + delta * tanh(z), so A_r(t) equals the anchor at
    initialisation and the per-sample correction is bounded by +-delta in each logit; delta = 0 fixes
    A_r(t) at the anchor. This keeps the input-dependent tables from drifting to the degenerate
    "teachers miss most spindles" solution seen in the GPU grid."""

    def __init__(self, c_in, n_teachers=N_TEACHERS, width=32, anchor=None, delta=1.0):
        super().__init__()
        layers, c = [], c_in
        for d in (1, 2, 4):
            layers += [nn.Conv1d(c, width, 7, padding="same", dilation=d, bias=False), nn.BatchNorm1d(width), nn.ReLU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv1d(width, n_teachers * 4, 1)
        self.R = n_teachers
        nn.init.zeros_(self.out.weight)
        self.delta = delta
        if anchor is not None:
            a = torch.as_tensor(anchor, dtype=torch.float32)
            assert a.shape == (n_teachers, 2, 2) and torch.allclose(a.sum(-1), torch.ones(n_teachers, 2))
            self.register_buffer("anchor_logits", torch.log(a.clamp_min(1e-6)))
            nn.init.zeros_(self.out.bias)
        else:
            self.anchor_logits = None
            with torch.no_grad():
                b = torch.zeros(n_teachers, 2, 2)
                b[:, 0, 0] = b[:, 1, 1] = math.log(DIAG_INIT / (1 - DIAG_INIT))
                self.out.bias.copy_(b.flatten())

    def forward(self, h):
        z = self.out(self.body(h))                      # [N, R*4, T]
        z = z.view(z.shape[0], self.R, 2, 2, z.shape[-1])
        if self.anchor_logits is not None:
            z = self.anchor_logits[None, :, :, :, None] + self.delta * torch.tanh(z)
        return torch.softmax(z, dim=3)                  # rows sum to 1


class GlobalNoise(nn.Module):
    """ablation: one 2x2 matrix per teacher, the same at every sample"""

    def __init__(self, n_teachers=N_TEACHERS):
        super().__init__()
        b = torch.zeros(n_teachers, 2, 2)
        b[:, 0, 0] = b[:, 1, 1] = math.log(DIAG_INIT / (1 - DIAG_INIT))
        self.logits = nn.Parameter(b)

    def forward(self, n, t):
        return torch.softmax(self.logits, dim=2)[None, :, :, :, None].expand(n, -1, -1, -1, t)


# ------------------------------------------------------------------ the student

class Student(nn.Module):
    def __init__(self, trunk="S", stream_b=True, pcen=True, noise="local", noise_trunk_features=True,
                 stream_b_into_trunk=False, boundary=True, bigru=False, prior=0.04, n_teachers=N_TEACHERS,
                 noise_anchor=None, anchor_delta=1.0):
        super().__init__()
        if stream_b_into_trunk and trunk != "M":
            raise ValueError("stream B into the trunk is only defined for trunk M (trunk S is SUMOv2's, unchanged)")
        self.trunk = TrunkS() if trunk == "S" else TrunkM(bigru=bigru)
        self.trunk_name = trunk
        self.stream_b = StreamB(pcen=pcen) if (stream_b and noise == "local") or stream_b_into_trunk else None
        self.stream_b_into_trunk = stream_b_into_trunk
        if stream_b_into_trunk:   # trunk M only: its stem takes 1 + 8 input channels
            self.trunk.stem = ConvBNReLU(1 + self.stream_b.n_channels, TrunkM.channels[0], TrunkM.stem_kernel)
        self.noise_kind = noise
        self.noise_trunk_features = noise_trunk_features
        c_noise = (self.stream_b.n_channels if self.stream_b is not None else 0) + \
                  (self.trunk.bottleneck_channels if noise_trunk_features else 0)
        if noise == "local":
            if c_noise == 0:
                raise ValueError("the local noise branch needs stream B or trunk features")
            self.noise = NoiseBranch(c_noise, n_teachers, anchor=noise_anchor, delta=anchor_delta)
        elif noise == "global":
            if noise_anchor is not None:
                raise ValueError("an anchor is only defined for the local noise branch")
            self.noise = GlobalNoise(n_teachers)
        else:
            self.noise = None
        self.boundary = nn.Conv1d(self.trunk.out_channels, 1, 1) if boundary else None
        self.init_priors(prior)

    def init_priors(self, prior, boundary_prior=0.05):
        """head 1 bias so that P(spindle) = prior at initialisation (SEED); trunk S's logits pass
        through ReLU, so the background logit carries the offset and the spindle logit starts at 0"""
        dense = self.trunk.dense[0] if isinstance(self.trunk.dense, nn.Sequential) else self.trunk.dense
        with torch.no_grad():
            dense.bias.copy_(torch.tensor([math.log((1 - prior) / prior), 0.0]))
            if self.boundary is not None:
                self.boundary.bias.fill_(math.log(boundary_prior / (1 - boundary_prior)))

    def _trunk(self, x):
        if self.stream_b_into_trunk:
            sb = self.stream_b(x)
            return (*self.trunk(torch.cat([x.unsqueeze(1), sb], 1)), sb)
        return (*self.trunk(x), None)

    def forward(self, x, need_noise=True):
        logits, feat, bott, off, sb = self._trunk(x)
        out = dict(logits=logits, p=torch.softmax(logits, 1)[:, 1])
        if self.boundary is not None:
            out["b_logit"] = self.boundary(feat)[:, 0]
        if self.noise is not None and need_noise:
            n, t = x.shape
            if self.noise_kind == "global":
                out["A"] = self.noise(n, t)
            else:
                parts = []
                if self.stream_b is not None:
                    parts.append(sb if sb is not None else self.stream_b(x))
                if self.noise_trunk_features:
                    # nearest up-sampling of the bottleneck to the padded length, then the same crop
                    up = F.interpolate(bott, scale_factor=self.trunk.stride, mode="nearest")
                    parts.append(up[:, :, off: off + t])
                out["A"] = self.noise(torch.cat(parts, 1))
        return out

    @torch.no_grad()
    def predict(self, x):
        """P(spindle) with the trunk's inference post-processing (trunk S: SUMOv2's moving average)"""
        logits, feat, _, _, _ = self._trunk(x)
        p = torch.softmax(self.trunk.postprocess(logits), 1)[:, 1]
        b = torch.sigmoid(self.boundary(feat)[:, 0]) if self.boundary is not None else None
        return p, b


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_table(model):
    """trainable parameters per component (for the documentation and the tests)"""
    out = {"trunk": count_parameters(model.trunk)}
    if model.stream_b is not None:
        out["stream_b"] = count_parameters(model.stream_b)
    if model.noise is not None:
        out["noise"] = count_parameters(model.noise)
    if model.boundary is not None:
        out["head2"] = count_parameters(model.boundary)
    out["total"] = count_parameters(model)
    return out
