"""
Row 4 feasibility audit (#3): Multi-channel coverage, contralateral C4->C3 faint-spindle recovery,
and occipital O2 alpha-veto rates across HMC and CinC 2018 (40 validation nights).

No expert labels read.

    python audit_multichannel.py

Outputs -> results/student/multichannel_audit/summary.csv
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import data
from spindlebench import paths
from spindlebench.data.cache import signal_path

OUT = data.STUDY / "multichannel_audit"


def band_rms_db(x, sf, lo, hi, n2_mask, win_s=0.5):
    """0.5-s smoothed bandpass RMS in dB relative to recording N2 median RMS."""
    b, a = butter(4, [lo / (sf / 2.0), hi / (sf / 2.0)], btype="band")
    xf = filtfilt(b, a, np.asarray(x, dtype=np.float64))
    sq = xf ** 2
    k = max(int(round(win_s * sf)), 1)
    ker = np.ones(k, dtype=np.float64) / k
    rms = np.sqrt(np.maximum(np.convolve(sq, ker, mode="same"), 1e-16))
    med = float(np.median(rms[n2_mask])) if n2_mask.any() else float(np.median(rms))
    return 20.0 * np.log10(rms / max(med, 1e-8)), rms


def downsample_to_100(arr, sf, n100):
    idx = np.minimum((np.arange(n100) * sf / data.FS).astype(int), len(arr) - 1)
    return arr[idx]


def runs_ge(mask, min_len):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (e - s) >= min_len
    return list(zip(s[keep], e[keep]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = data.manifest()
    sp = data.splits()

    # 1. Channel coverage across all 401 nights
    cov = {}
    for ds, g in man.groupby("dataset"):
        has_all = 0
        for r in g.itertuples():
            z = np.load(signal_path(r.path, paths.CACHE / "manifest.csv"))
            chs = {str(c) for c in z["ch_names"]}
            if {"C3-M2", "C4-M1", "F4-M1", "O2-M1"}.issubset(chs):
                has_all += 1
        cov[ds] = (has_all, len(g))
        print(f"Channel coverage {ds:8s}: {has_all}/{len(g)} nights have C3-M2, C4-M1, F4-M1, O2-M1", flush=True)

    # 2. Contralateral C4 recovery & occipital O2 alpha-veto on the 40 validation nights
    val = sp[sp.role == "val"]
    man_idx = man.set_index(["dataset", "rec"])
    rows = []

    for ds, rec in zip(val.dataset, val.rec):
        r = man_idx.loc[(ds, rec)]
        z = np.load(signal_path(r.path, paths.CACHE / "manifest.csv"))
        ch = [str(c) for c in z["ch_names"]]
        if not {"C3-M2", "C4-M1", "O2-M1"}.issubset(ch):
            continue
        sf = float(z["sf"])
        _, code = data.open_night(ds, rec)
        code = np.asarray(code)
        n100 = len(code)
        valid = (code & data.VALID_BIT) > 0
        pat = code & (data.VALID_BIT - 1)
        n_votes = sum((pat >> i) & 1 for i in range(len(data.TEACHERS)))

        # Native-rate N2 mask for baseline median
        idx_nat = np.minimum((np.arange(z["data"].shape[-1]) * data.FS / sf).astype(int), n100 - 1)
        n2_nat = valid[idx_nat]

        c3_sig_db, _ = band_rms_db(z["data"][ch.index("C3-M2")], sf, 11.0, 16.0, n2_nat)
        c4_sig_db, _ = band_rms_db(z["data"][ch.index("C4-M1")], sf, 11.0, 16.0, n2_nat)
        o2_alp_db, o2_alp_rms = band_rms_db(z["data"][ch.index("O2-M1")], sf, 8.0, 10.5, n2_nat)
        _, o2_sig_rms = band_rms_db(z["data"][ch.index("O2-M1")], sf, 11.0, 16.0, n2_nat)

        c3_db = downsample_to_100(c3_sig_db, sf, n100)
        c4_db = downsample_to_100(c4_sig_db, sf, n100)
        o2_db = downsample_to_100(o2_alp_db, sf, n100)
        o2_alp_dom = downsample_to_100(o2_alp_rms > 1.5 * o2_sig_rms, sf, n100)

        n2_min = valid.sum() / data.FS / 60.0

        # Check A: C4 >= 9 dB sustained >= 0.4 s while C3 has 0/4 teachers
        c4_clear_c3_silent = runs_ge(valid & (n_votes == 0) & (c4_db >= 9.0), 40)
        c3_snrs_at_c4 = [float(np.median(c3_db[a:b])) for a, b in c4_clear_c3_silent]
        faint_c3_hits = sum(3.0 <= s <= 8.0 for s in c3_snrs_at_c4)

        # Check B: C3 majority-vote events (>= 2/4) vs O2 alpha & C4 support
        mv_evs = runs_ge(valid & (n_votes >= 2), 30)
        n_mv = len(mv_evs)
        n_o2_alpha = sum(1 for a, b in mv_evs if (o2_db[a:b].mean() >= 6.0 and o2_alp_dom[a:b].mean() >= 0.5))
        n_unilat = sum(1 for a, b in mv_evs if c4_db[a:b].max() < 3.0)

        rows.append(dict(
            dataset=ds, rec=rec, n2_min=n2_min,
            c3_mv_per_min=n_mv / n2_min,
            c4_clear_c3_silent_per_min=len(c4_clear_c3_silent) / n2_min,
            c4_clear_c3_faint3to8_per_min=faint_c3_hits / n2_min,
            c3_snr_median_when_c4_clear=float(np.median(c3_snrs_at_c4)) if c3_snrs_at_c4 else np.nan,
            mv_o2_alpha_frac=n_o2_alpha / max(n_mv, 1),
            mv_unilateral_c4_low_frac=n_unilat / max(n_mv, 1),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    print("\n=== Row 4 Multi-Channel Audit (40 validation nights, clean N2) ===")
    print(df.groupby("dataset").mean(numeric_only=True).round(3).to_string())
    print("\nOverall mean across 40 nights:")
    print(df.mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
