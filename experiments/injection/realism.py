"""
Realism check (plan section 2): label-free features of real expert spindles vs injected spindles,
measured the same way on the same recordings.

    .venv/Scripts/python.exe injection/realism.py
-> results/injection/realism/features.csv, summary.csv, realism.png
"""

import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.data.cache import signal_path
from spindlebench.synth.generator import Params, eligible_mask, inject, sigma_band
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

OUT = paths.RESULTS / "injection" / "realism"
NAP_HALF = 0.5   # NAP marks are points near the spindle centre


def features(x, sig, sf, on, du):
    """features of one event window [on, on+du]"""
    n = len(x)
    a, b = int(on * sf), int((on + du) * sf)
    if a < 0 or b > n or b - a < 4:
        return None
    s = sig[a:b]
    w = int(5 * sf)
    ctx = np.r_[sig[max(a - w, 0):a], sig[b:min(b + w, n)]]
    seg = x[a:b] - x[a:b].mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=int(8 * sf))) ** 2
    fr = np.fft.rfftfreq(int(8 * sf), 1 / sf)
    band = (fr >= 9) & (fr <= 17)
    broad = (fr >= 0.5) & (fr <= 30)
    sb = (fr >= 11) & (fr <= 16)
    return dict(
        amp_ptp=float(s.max() - s.min()),
        peak_freq=float(fr[band][np.argmax(spec[band])]),
        snr_db=float(20 * np.log10(np.sqrt(np.mean(s ** 2)) / max(np.sqrt(np.mean(ctx ** 2)), 1e-9))) if len(ctx) else np.nan,
        rel_sigma=float(spec[sb].sum() / max(spec[broad].sum(), 1e-12)),
        duration=du,
    )


def run():
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    m = m[m.dataset.isin(["dreams_spindles", "nap"]) & ~m.excluded.astype(bool)]
    p = Params()
    rows, meta = [], []
    for seed, r in enumerate(m.itertuples()):
        z = np.load(signal_path(r.path))
        ch = [str(c) for c in z["ch_names"]]
        x = z["data"][ch.index(r.default_channel)].astype(float)
        sf = float(z["sf"])
        n = len(x)
        elig, n2, used = eligible_mask(r.dataset, r.rec, r.default_channel, n, sf, p)
        xi, ev = inject(x, sf, elig, p, seed=1000 + seed)
        sig, sigi = sigma_band(x, sf), sigma_band(xi, sf)
        t_n2 = lambda t: n2[min(int(t * sf), n - 1)]  # noqa: E731
        if r.dataset == "dreams_spindles":
            lab = pd.read_csv(paths.CACHE / "labels" / r.dataset / f"{r.rec}__expert1__spindle.csv")
            lab = lab[(lab.onset + lab.duration <= 1000)]
            real = [(o, d) for o, d in zip(lab.onset, lab.duration) if t_n2(o + d / 2)]
        else:
            lab = pd.read_csv(paths.CACHE / "labels" / r.dataset / f"{r.rec}__expert__spindle.csv")
            real = [(o - NAP_HALF, 2 * NAP_HALF) for o in lab.onset if t_n2(o)]
        for o, d in real:
            f = features(x, sig, sf, o, d)
            if f:
                rows.append(dict(dataset=r.dataset, rec=r.rec, kind="real", **f))
        for e in ev.itertuples():
            if r.dataset == "nap":   # measure like a NAP mark: centre +- 0.5 s
                o, d = e.onset + e.duration / 2 - NAP_HALF, 2 * NAP_HALF
            else:
                o, d = e.onset, e.duration
            f = features(xi, sigi, sf, o, d)
            if f:
                rows.append(dict(dataset=r.dataset, rec=r.rec, kind="injected", target_snr=e.snr_db, **f))
        meta.append(dict(dataset=r.dataset, rec=r.rec, sf=sf, n2_min=n2.sum() / sf / 60,
                         quiet_n2_min=elig.sum() / sf / 60, n_injected=len(ev), n_real=len(real),
                         quiet_detectors="+".join(used)))
        print(f"{r.dataset:16s} {r.rec:10s} N2 {n2.sum()/sf/60:5.1f} min, quiet {elig.sum()/sf/60:5.1f} min, "
              f"injected {len(ev):3d}, real {len(real):3d}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame(rows)
    f.to_csv(OUT / "features.csv", index=False)
    pd.DataFrame(meta).to_csv(OUT / "recordings.csv", index=False)
    q = (f.groupby(["dataset", "kind"])[["amp_ptp", "peak_freq", "snr_db", "rel_sigma", "duration"]]
         .quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3))
    q.to_csv(OUT / "summary.csv")
    print(q.to_string())
    plot(f)


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats = [("amp_ptp", "Sigma-band peak-to-peak (µV)"), ("peak_freq", "Peak frequency (Hz)"),
             ("snr_db", "Local sigma SNR (dB)"), ("rel_sigma", "Relative sigma power")]
    colors = {"real": "#1F4E79", "injected": "#E07B39"}
    fig, ax = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for i, ds in enumerate(["dreams_spindles", "nap"]):
        for j, (c, lab) in enumerate(feats):
            a = ax[i, j]
            d = f[f.dataset == ds]
            lo, hi = np.nanpercentile(d[c], [1, 99])
            bins = np.linspace(lo, hi, 40)
            for k in ["real", "injected"]:
                v = d[d.kind == k][c].dropna()
                a.hist(v, bins=bins, density=True, histtype="step", lw=2, color=colors[k], label=f"{k} (n={len(v)})")
            a.set_title(f"{'DREAMS' if ds.startswith('dreams') else 'NAP'}: {lab}", fontsize=10)
            a.spines[["top", "right"]].set_visible(False)
            a.set_yticks([])
            if j == 0:
                a.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "realism.png", dpi=130)


if __name__ == "__main__":
    run()
