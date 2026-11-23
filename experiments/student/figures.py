"""
Figures for docs/student_architecture.md.

    uv run python figures.py          (from experiments/student)

-> docs/figures/student/stream_b_init.png: stream B at initialisation.
   (a) magnitude response of the 8 Gabor filters, the 11-16 Hz sigma band shaded;
   (b) a 40-s training crop (HMC, valid N2) with the majority-vote consensus events shaded;
   (c) the envelope E_k(t) of the 12.5 Hz channel; (d) its sPCEN output P_k(t).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import data  # noqa: E402
import label_models  # noqa: E402
import model as M  # noqa: E402
from spindlebench import paths, provenance  # noqa: E402

OUT = paths.ROOT / "docs" / "figures" / "student"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROLE = {"in band": "#2a78d6", "slow edge": "#eb6834", "contrast": "#1baf7a"}
ROLES = ["in band"] * 4 + ["slow edge"] * 2 + ["contrast"] * 2


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


def pick_crop():
    """first 40-s valid window of SN001 with >= 3 consensus events, all inside the central 30 s"""
    x, code = data.open_night("hmc", "SN001")
    code = np.asarray(code)
    valid = (code & data.VALID_BIT) > 0
    mv = label_models.load_targets()["mv"][code & (data.VALID_BIT - 1)] > 0.5
    for a in range(0, len(code) - data.CROP, 500):
        if valid[a:a + data.CROP].all():
            s, e = data.mask_to_events(mv[a:a + data.CROP]).T if mv[a:a + data.CROP].any() else ([], [])
            if len(s) >= 3 and s.min() >= data.CORE_LO and e.max() <= data.CORE_LO + data.CORE:
                return np.asarray(x[a:a + data.CROP], np.float32), mv[a:a + data.CROP], a
    raise RuntimeError("no crop")


def main():
    provenance.track(__file__, protocols=[])
    OUT.mkdir(parents=True, exist_ok=True)
    sb = M.StreamB()
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), gridspec_kw=dict(height_ratios=[1.3, 1, 1, 1]))
    fig.patch.set_facecolor(SURFACE)

    # (a) frequency responses
    ax = axes[0]
    style(ax)
    re, im = sb.gabor()
    g = (re + 1j * im)[:, 0].detach().numpy()
    nfft = 4096
    H = np.abs(np.fft.fft(g, nfft, axis=1))[:, : nfft // 2]
    f = np.fft.fftfreq(nfft, 1 / data.FS)[: nfft // 2]
    ax.axvspan(11, 16, color=GRID, lw=0)
    ax.text(13.5, 1.245, "sigma band 11–16 Hz", ha="center", va="bottom", fontsize=7.5, color=INK2)
    seen = set()
    for k, fk in enumerate(sb.centre_frequencies().detach().numpy()):
        lab = ROLES[k] if ROLES[k] not in seen else None
        seen.add(ROLES[k])
        ax.plot(f, H[k], color=ROLE[ROLES[k]], lw=2, label=lab, solid_capstyle="round")
        ax.text(fk, H[k].max() + (0.03 if k % 2 == 0 else 0.13), f"{fk:g} Hz", ha="center", va="bottom", fontsize=7,
                color=INK)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.36)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|G_k(f)|")
    ax.set_title("(a) Gabor filters at initialisation (σ = 0.12 s)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2, bbox_to_anchor=(1.0, 0.9))

    # (b)-(d) a real crop
    x, mv, a0 = pick_crop()
    t = np.arange(len(x)) / data.FS
    with torch.no_grad():
        xt = torch.from_numpy(x)[None]
        E = sb.envelope(xt)[0, 1].numpy()
        P = sb(xt)[0, 1].numpy()
    runs = data.mask_to_events(mv)
    for ax, y, ttl, yl in ((axes[1], x, "(b) EEG crop, HMC SN001 (robust-scaled), consensus events shaded", "x(t)"),
                           (axes[2], E, "(c) envelope E_k(t), 12.5 Hz channel", "E_k(t)"),
                           (axes[3], P, "(d) sPCEN output P_k(t), 12.5 Hz channel", "P_k(t)")):
        style(ax)
        for s, e in runs:
            ax.axvspan(s / data.FS, e / data.FS, color=GRID, lw=0)
        ax.plot(t, y, color=ROLE["in band"], lw=1 if y is x else 2)
        ax.set_xlim(0, t[-1])
        ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left", fontsize=10, color=INK)
    axes[3].set_xlabel(f"time in crop (s); crop starts at {a0 / data.FS:.0f} s")
    fig.tight_layout()
    fig.savefig(OUT / "stream_b_init.png", dpi=150, facecolor=SURFACE)
    print(OUT / "stream_b_init.png")


if __name__ == "__main__":
    main()
