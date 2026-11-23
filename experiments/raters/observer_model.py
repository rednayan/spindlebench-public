"""
Core analysis (#1 from next_steps.md): Psychophysical observer model for DREAMS and NAP.

Fits each expert's detection curve as a lapse-limited sigmoid of local sigma SNR (dB):
    P(expert marks | SNR = s) = gamma + (1 - gamma - lam) * sigmoid(k * (s - s0))

where:
    s0       = 50%-point threshold (dB)
    k        = slope (dB^-1)
    lam      = lapse rate (fraction of arbitrarily strong spindles missed by the expert)
    1 - lam  = asymptotic plateau (caps precision of even a perfect detector)
    F1_ceil  = 2*(1 - lam) / (2 - lam)  (implied F1 ceiling at recall = 1.0)

Fitted independently using 4 diverse detectors (SUMOv2, A7, teachers' majority vote, and the
headline student) to verify that the inferred expert curve is detector-independent.

    python experiments/raters/observer_model.py

Outputs -> results/observer_model/
    fits.csv              fitted parameters + cluster-bootstrap 95% CIs (by recording / person)
    binned_curves.csv     empirical P(expert marks) in SNR bins
    observer_curves.png   publication-ready 2-panel psychophysical curve figure
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spindlebench import paths
from spindlebench.data import nap as nap_data
from spindlebench.data.cache import labels_dir, signal_path
from spindlebench.metrics import events as em

OUT = paths.RESULTS / "observer_model"
BOOT = 400
DETECTORS = [
    ("sumov2", "SUMOv2 (learned)", [paths.RESULTS / "detections"]),
    ("a7", "A7 (rule)", [paths.RESULTS / "detections_clean", paths.RESULTS / "detections"]),
    ("consensus_mv", "Teachers' majority vote", [paths.RESULTS / "student" / "detections"]),
    ("student_rules4_fixA_gpu_s0", "Headline student (fixA s0)", [paths.RESULTS / "student" / "detections"]),
]


def bandpass_sigma(x, sf, lo=11.0, hi=16.0):
    b, a = butter(4, [lo / (sf / 2.0), hi / (sf / 2.0)], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64))


def local_sigma_snr_db(x_sig, sf, onsets, offsets, n2_mask=None):
    """Local sigma SNR (dB) for each event [onset, offset] against flanking +-15 s background
    (excluding +-1 s around the event)."""
    n = len(x_sig)
    sq = x_sig ** 2
    # Recording-level N2 fallback RMS
    bg_all = np.sqrt(np.median(sq[n2_mask])) if (n2_mask is not None and n2_mask.any()) else np.sqrt(np.median(sq))
    bg_all = max(float(bg_all), 1e-8)

    snrs = np.empty(len(onsets), dtype=float)
    win = int(round(15.0 * sf))
    guard = int(round(1.0 * sf))
    for k, (o, e) in enumerate(zip(onsets, offsets)):
        i0 = max(int(round(o * sf)), 0)
        i1 = min(int(round(e * sf)), n)
        if i1 <= i0:
            snrs[k] = np.nan
            continue
        ev_rms = np.sqrt(float(np.mean(sq[i0:i1])))
        l0, l1 = max(0, i0 - win), max(0, i0 - guard)
        r0, r1 = min(n, i1 + guard), min(n, i1 + win)
        flank = np.r_[sq[l0:l1], sq[r0:r1]]
        if len(flank) >= int(2 * sf):
            # Robust local background RMS using median of 0.5-s RMS blocks
            blk = max(int(round(0.5 * sf)), 1)
            m = (len(flank) // blk) * blk
            bg_rms = float(np.median(np.sqrt(flank[:m].reshape(-1, blk).mean(axis=1))))
        else:
            bg_rms = bg_all
        bg_rms = max(bg_rms, 1e-8)
        snrs[k] = 20.0 * np.log10(max(ev_rms, 1e-12) / bg_rms)
    return snrs


def load_det(det_name, stores, ds, rec, channel):
    for st in stores:
        f = st / ds / det_name / f"{rec}.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"channel": str})
            d = d[(d.event == "spindle") & (d.channel == str(channel))]
            on = d.onset.to_numpy(float)
            du = np.nan_to_num(d.duration.to_numpy(float))
            return on, on + du
    return None, None


def collect_dreams(det_name, stores, expert="expert1"):
    man = pd.read_csv(paths.CACHE / "manifest.csv")
    man = man[man.dataset == "dreams_spindles"].reset_index(drop=True)
    rows = []
    for r in man.itertuples():
        ref_f = labels_dir() / "dreams_spindles" / f"{r.rec}__{expert}__spindle.csv"
        if not ref_f.exists():
            continue
        z = np.load(signal_path(r.path, paths.CACHE / "manifest.csv"))
        ch = [str(c) for c in z["ch_names"]]
        x = z["data"][ch.index(str(r.default_channel))].astype(float)
        sf = float(z["sf"])
        x_sig = bandpass_sigma(x, sf)

        # N2 mask + first 1000 s (DREAMS primary protocol)
        h = pd.read_csv(labels_dir() / "dreams_spindles" / f"{r.rec}__hypno.csv")
        ep = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
        idx = np.clip((np.arange(len(x)) / sf // ep).astype(int), 0, len(h) - 1)
        n2 = (h.stage.to_numpy().astype(str)[idx] == "N2") & (np.arange(len(x)) / sf < 1000.0)

        on, off = load_det(det_name, stores, "dreams_spindles", r.rec, r.default_channel)
        if on is None:
            continue
        c = (on + off) / 2.0
        keep = (on < 1000.0) & n2[np.clip(np.round(c * sf).astype(int), 0, len(x) - 1)]
        on, off = on[keep], off[keep]
        if not len(on):
            continue

        ref = pd.read_csv(ref_f)
        ron = ref.onset.to_numpy(float)
        rdu = np.nan_to_num(ref.duration.to_numpy(float), nan=1.0)
        rc = ron + rdu / 2.0
        rkeep = (ron < 1000.0) & n2[np.clip(np.round(rc * sf).astype(int), 0, len(x) - 1)]
        ron, rdu = ron[rkeep], rdu[rkeep]

        _, _, _, pairs = em.match(np.c_[on, off - on], np.c_[ron, rdu], "onset", 0.5)
        hit = np.zeros(len(on), dtype=int)
        for pi, _ in pairs:
            hit[pi] = 1
        snrs = local_sigma_snr_db(x_sig, sf, on, off, n2_mask=n2)
        for s, y in zip(snrs, hit):
            if np.isfinite(s):
                rows.append(dict(cluster=r.rec, snr_db=s, hit=y))
    return pd.DataFrame(rows)


def collect_nap(det_name, stores):
    man = pd.read_csv(paths.CACHE / "manifest.csv")
    man = man[man.dataset == "nap"].set_index("rec")
    infos = [nap_data.nap_info(r) for r in nap_data.naps().itertuples()]
    rows = []
    for info in infos:
        rec = info["rec"]
        r = man.loc[rec]
        z = np.load(signal_path(r.path, paths.CACHE / "manifest.csv"))
        ch = [str(c) for c in z["ch_names"]]
        x = z["data"][ch.index("C3")].astype(float)
        sf = float(z["sf"])
        x_sig = bandpass_sigma(x, sf)

        on, off = load_det(det_name, stores, "nap", rec, "C3")
        if on is None:
            continue
        c = (on + off) / 2.0
        keep = nap_data.in_n2(info, c) if len(on) else np.zeros(0, bool)
        on, off, c = on[keep], off[keep], c[keep]
        if not len(on):
            continue

        marks = np.asarray(info["marks"], float)
        _, _, _, pairs = em.match(np.c_[c, np.zeros(len(c))], np.c_[marks, np.zeros(len(marks))], "onset", 0.5)
        hit = np.zeros(len(on), dtype=int)
        for pi, _ in pairs:
            hit[pi] = 1
        t_grid = np.arange(len(x)) / sf
        n2_mask = nap_data.in_n2(info, t_grid)
        snrs = local_sigma_snr_db(x_sig, sf, on, off, n2_mask=n2_mask)
        for s, y in zip(snrs, hit):
            if np.isfinite(s):
                rows.append(dict(cluster=info["subject"], snr_db=s, hit=y))
    return pd.DataFrame(rows)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def psych_prob(s, s0, k, lam, gamma):
    return np.clip(gamma + (1.0 - gamma - lam) * sigmoid(k * (s - s0)), 1e-6, 1.0 - 1e-6)


def fit_observer(snr, hit):
    """MLE fit of lapse-limited sigmoid: params = (s0, k, lam, gamma)."""
    s = np.asarray(snr, float)
    y = np.asarray(hit, float)

    def nll(theta):
        s0, k, lam, gamma = theta
        if lam + gamma >= 0.99:
            return 1e9
        p = psych_prob(s, s0, k, lam, gamma)
        return -float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    best = None
    bounds = [(-5.0, 20.0), (0.05, 5.0), (0.0, 0.95), (0.0, 0.25)]
    for s0_init in (4.0, 7.0, 10.0):
        for lam_init in (0.1, 0.45):
            res = minimize(nll, [s0_init, 0.5, lam_init, 0.02], method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
    s0, k, lam, gamma = best.x
    plateau = 1.0 - lam
    p_9dB = float(psych_prob(9.0, s0, k, lam, gamma))
    p_12dB = float(psych_prob(12.0, s0, k, lam, gamma))
    f1_ceil = 2.0 * plateau / (1.0 + plateau)
    return dict(s0=float(s0), k=float(k), lam=float(lam), gamma=float(gamma),
                plateau=float(plateau), p_9dB=p_9dB, p_12dB=p_12dB, f1_ceil=float(f1_ceil))


def fit_with_bootstrap(df, seed=0):
    base = fit_observer(df.snr_db, df.hit)
    clusters = df.cluster.unique()
    by_c = {c: g for c, g in df.groupby("cluster")}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(BOOT):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        sub = pd.concat([by_c[c] for c in draw], ignore_index=True)
        boots.append(fit_observer(sub.snr_db, sub.hit))
    bdf = pd.DataFrame(boots)
    out = dict(n_events=len(df), n_clusters=len(clusters), **base)
    for col in ("s0", "k", "lam", "plateau", "f1_ceil"):
        out[f"{col}_lo"], out[f"{col}_hi"] = np.percentile(bdf[col], [2.5, 97.5])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("dreams_e1", "DREAMS Expert 1", lambda n, s: collect_dreams(n, s, "expert1")),
        ("dreams_e2", "DREAMS Expert 2", lambda n, s: collect_dreams(n, s, "expert2")),
        ("nap", "NAP Single Rater", lambda n, s: collect_nap(n, s)),
    ]

    fit_rows, bin_rows = [], []
    bins = np.array([-2, 2, 4, 6, 8, 10, 12, 15, 25], dtype=float)

    for ds_key, ds_label, loader in datasets:
        for det_key, det_label, stores in DETECTORS:
            df = loader(det_key, stores)
            if df.empty:
                continue
            res = fit_with_bootstrap(df, seed=2026)
            fit_rows.append(dict(dataset=ds_key, dataset_label=ds_label,
                                 detector=det_key, detector_label=det_label, **res))
            print(f"{ds_label:16s} | {det_label:26s} | n={res['n_events']:4d} | "
                  f"s0={res['s0']:5.2f} dB | plateau={res['plateau']:.3f} [{res['plateau_lo']:.3f},{res['plateau_hi']:.3f}] | "
                  f"lapse={res['lam']:.3f} | F1_ceil={res['f1_ceil']:.3f}", flush=True)
            df["bin"] = pd.cut(df["snr_db"], bins)
            for b, g in df.groupby("bin", observed=True):
                if len(g) >= 5:
                    bin_rows.append(dict(dataset=ds_key, detector=det_key,
                                         snr_mid=float(g["snr_db"].mean()),
                                         p_emp=float(g["hit"].mean()), n=len(g)))

    fdf = pd.DataFrame(fit_rows)
    bdf = pd.DataFrame(bin_rows)
    fdf.to_csv(OUT / "fits.csv", index=False)
    bdf.to_csv(OUT / "binned_curves.csv", index=False)

    # Plot 3-panel psychophysical observer curves
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=200, sharey=True)
    colors = {"sumov2": "#1f77b4", "a7": "#2ca02c", "consensus_mv": "#ff7f0e",
              "student_rules4_fixA_gpu_s0": "#d62728"}
    s_grid = np.linspace(0, 18, 200)

    for ax, (ds_key, ds_label, _) in zip(axes, datasets):
        sub_f = fdf[fdf.dataset == ds_key]
        sub_b = bdf[bdf.dataset == ds_key]
        for row in sub_f.itertuples():
            col = colors[row.detector]
            p = psych_prob(s_grid, row.s0, row.k, row.lam, row.gamma)
            ax.plot(s_grid, p, lw=2.0, color=col,
                    label=f"{row.detector_label} (1−λ={row.plateau:.2f})")
            gb = sub_b[sub_b.detector == row.detector]
            ax.scatter(gb.snr_mid, gb.p_emp, s=np.clip(gb.n * 0.4, 15, 70), color=col, alpha=0.6)
        med_p9 = sub_f["p_9dB"].median()
        med_p12 = sub_f["p_12dB"].median()
        med_s0 = sub_f["s0"].median()
        ax.axvline(9.0, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.axvline(12.0, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.set_title(f"{ds_label}\nP(mark | 9 dB) = {med_p9:.2f}  |  P(mark | 12 dB) = {med_p12:.2f}  |  s0 ≈ {med_s0:.1f} dB",
                     fontsize=9.5)
        ax.set_xlabel("Local sigma SNR (dB)")
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    axes[0].set_ylabel("P(expert marks detected burst)")
    fig.suptitle("Psychophysical Observer Model: Expert Marking Probability vs. Burst Sigma SNR\n"
                 "(Consistent plateau across 4 independent detectors isolates rater lapse rate λ)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "observer_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
