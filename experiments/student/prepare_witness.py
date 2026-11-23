"""
Precompute compact 1-byte/sample multi-channel witness masks (Row 4) for all 401 HMC + CinC 2018
nights using independent contralateral C4-M1 and occipital O2-M1 channels.

No expert labels are read. All 401/401 nights have C3-M2, C4-M1, and O2-M1 with independent
contralateral mastoid references (M2 vs M1).

Bit layout of <rec>__mc.npy (uint8, 100 Hz grid, aligned with arrays/<ds>/<rec>__code.npy):
    bit 0 (1): c4_clear    C4-M1 sigma >= 9 dB (sustained >= 0.4 s) & sigma > alpha & sigma > beta
    bit 1 (2): c4_support  C4-M1 sigma >= 3 dB (bilateral sigma support)
    bit 2 (4): o2_alpha    O2-M1 alpha >= 6 dB & alpha > 1.5 * sigma (occipital alpha veto)
    bit 3 (8): c3_faint    C3-M2 sigma >= 3 dB & sigma > alpha (faint sigma actually present on C3)

    python prepare_witness.py [--workers 8]

Outputs -> results/student/work/witness/<ds>/<rec>__mc.npy, witness_summary.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import data
from spindlebench import paths
from spindlebench.data.cache import signal_path

WITNESS_DIR = data.WORK / "witness"
BIT_C4_CLEAR = 1
BIT_C4_SUPPORT = 2
BIT_O2_ALPHA = 4
BIT_C3_FAINT = 8


def _band_rms(x, sf, lo, hi, win_s=0.5):
    b, a = butter(4, [lo / (sf / 2.0), hi / (sf / 2.0)], btype="band")
    xf = filtfilt(b, a, np.asarray(x, dtype=np.float64))
    k = max(int(round(win_s * sf)), 1)
    ker = np.ones(k, dtype=np.float64) / k
    return np.sqrt(np.maximum(np.convolve(xf ** 2, ker, mode="same"), 1e-16))


def _to_db(rms, n2_mask):
    med = float(np.median(rms[n2_mask])) if n2_mask.any() else float(np.median(rms))
    return 20.0 * np.log10(rms / max(med, 1e-8))


def _keep_sustained(mask, min_len=40):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros(len(mask), dtype=bool)
    for a, b in zip(s, e):
        if b - a >= min_len:
            out[a:b] = True
    return out


def _ds100(arr, sf, n100):
    idx = np.minimum((np.arange(n100) * sf / data.FS).astype(int), len(arr) - 1)
    return arr[idx]


def process_one(args):
    ds, rec, rel_path = args
    data.guard(ds)
    out_f = WITNESS_DIR / ds / f"{rec}__mc.npy"
    _, code = data.open_night(ds, rec)
    code = np.asarray(code)
    n100 = len(code)
    valid = (code & data.VALID_BIT) > 0
    pat = code & (data.VALID_BIT - 1)
    n_votes = sum((pat >> i) & 1 for i in range(len(data.TEACHERS)))

    if out_f.exists():
        mc = np.load(out_f)
    else:
        z = np.load(signal_path(rel_path, paths.CACHE / "manifest.csv"))
        ch = [str(c) for c in z["ch_names"]]
        sf = float(z["sf"])
        idx_nat = np.minimum((np.arange(z["data"].shape[-1]) * data.FS / sf).astype(int), n100 - 1)
        n2_nat = valid[idx_nat]

        c3 = z["data"][ch.index("C3-M2")]
        c4 = z["data"][ch.index("C4-M1")]
        o2 = z["data"][ch.index("O2-M1")]

        c3_sig = _band_rms(c3, sf, 11.0, 16.0)
        c3_alp = _band_rms(c3, sf, 8.0, 10.5)
        c4_sig = _band_rms(c4, sf, 11.0, 16.0)
        c4_alp = _band_rms(c4, sf, 8.0, 10.5)
        c4_bet = _band_rms(c4, sf, 18.0, 30.0)
        o2_alp = _band_rms(o2, sf, 8.0, 10.5)
        o2_sig = _band_rms(o2, sf, 11.0, 16.0)

        c3_sig_db = _ds100(_to_db(c3_sig, n2_nat), sf, n100)
        c4_sig_db = _ds100(_to_db(c4_sig, n2_nat), sf, n100)
        o2_alp_db = _ds100(_to_db(o2_alp, n2_nat), sf, n100)

        c4_clear_raw = (c4_sig_db >= 9.0) & _ds100((c4_sig > c4_alp) & (c4_sig > c4_bet), sf, n100)
        c4_clear = _keep_sustained(c4_clear_raw, min_len=40)
        c4_support = c4_sig_db >= 3.0
        o2_alpha = (o2_alp_db >= 6.0) & _ds100(o2_alp > 1.5 * o2_sig, sf, n100)
        c3_faint = (c3_sig_db >= 3.0) & _ds100(c3_sig > c3_alp, sf, n100)

        mc = (
            (c4_clear.astype(np.uint8) * BIT_C4_CLEAR)
            | (c4_support.astype(np.uint8) * BIT_C4_SUPPORT)
            | (o2_alpha.astype(np.uint8) * BIT_O2_ALPHA)
            | (c3_faint.astype(np.uint8) * BIT_C3_FAINT)
        )
        out_f.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_f, mc)

    recov = valid & (n_votes == 0) & ((mc & BIT_C4_CLEAR) > 0) & ((mc & BIT_C3_FAINT) > 0) & ((mc & BIT_O2_ALPHA) == 0)
    veto = valid & (n_votes >= 2) & (((mc & BIT_O2_ALPHA) > 0) | ((mc & BIT_C4_SUPPORT) == 0))
    mv_s = (valid & (n_votes >= 2)).sum() / data.FS
    return dict(
        dataset=ds, rec=rec,
        valid_min=valid.sum() / data.FS / 60.0,
        mv_s=mv_s,
        recov_s=recov.sum() / data.FS,
        veto_frac=veto.sum() / max((valid & (n_votes >= 2)).sum(), 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    man = data.manifest()
    jobs = [(r.dataset, r.rec, r.path) for r in man.itertuples()]
    print(f"Precomputing multi-channel witness masks for {len(jobs)} nights ({a.workers} workers)...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(process_one, jobs), 1):
            rows.append(res)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(WITNESS_DIR / "witness_summary.csv", index=False)
    print("\n=== Multi-Channel Witness Summary (401 nights) ===")
    print(df.groupby("dataset")[["valid_min", "mv_s", "recov_s", "veto_frac"]].mean().round(3).to_string())
    tot_mv_h = df["mv_s"].sum() / 3600.0
    tot_recov_h = df["recov_s"].sum() / 3600.0
    print(f"\nTotal MV spindle duration: {tot_mv_h:.2f} h | Recovered C4-witnessed faint C3 duration: {tot_recov_h:.2f} h "
          f"(+{100*tot_recov_h/max(tot_mv_h,1e-6):.1f}%)")


if __name__ == "__main__":
    main()
