"""
Shared artefact rejection: one mask per recording, read by every detector and by the student.

    uv run spindlebench artefacts [--datasets hmc,cinc2018,...] [--recs ...]

Method (documented, fixed in advance; notes/plans/student_architecture.md, decision 2026-11-03):
YASA `art_detect` (Vallat & Walker 2021, yasa 0.8.0), `method="std"` for single-channel EEG: the
standard deviation of each 5-s window is log-transformed and z-scored separately within each sleep
stage of the public / expert hypnogram; |z| > 3 marks the window as an artefact. All YASA defaults
except `include`, which is widened to wake (0-4) so that wake is screened too.

Input: the recording's default channel from the cache, in µV at its native rate, after a 0.3 Hz
high-pass (2nd-order Butterworth, forward-backward) so that slow drifts do not inflate a window's SD;
the same high-pass every learned adapter in the benchmark applies first. Windows in epochs scored
`?` (unscored) are not tested and are written with `stage = "?"`, `artefact = 0`.

Rejection, not removal: the signal is never altered. Detectors run on the clean windows only;
the student leaves flagged windows out of its loss.

Output: cache/labels/<dataset>/<rec>__artefact.csv with one row per 5-s window:
    start, end, stage, zscore, artefact (0/1)
and results/artefacts/summary.csv, results/artefacts/per_recording.csv.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from spindlebench import paths
from spindlebench.data.cache import labels_dir, signal_path

VERSION = "artefacts.v1"
WINDOW = 5.0
THRESHOLD = 3.0
HIGHPASS = 0.3
STAGE_CODE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
INCLUDE = (0, 1, 2, 3, 4)
DATASETS = ["dreams_spindles", "nap", "boas", "hmc", "cinc2018"]


def hypno_per_sample(dataset, rec, n, sf):
    """integer hypnogram upsampled to the signal (YASA mapping; unscored = -2)"""
    h = pd.read_csv(labels_dir() / dataset / f"{rec}__hypno.csv")
    epoch = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    codes = h.stage.map(STAGE_CODE).fillna(-2).astype(int).to_numpy()
    idx = np.clip((np.arange(n) / sf // epoch).astype(int), 0, len(codes) - 1)
    return codes[idx]


def detect(x, sf, hyp):
    """-> DataFrame of 5-s windows (start, end, stage, zscore, artefact)"""
    import yasa
    x = sosfiltfilt(butter(2, HIGHPASS, "highpass", fs=sf, output="sos"), x)
    n_win = int(len(x) // (WINDOW * sf))
    n = int(n_win * WINDOW * sf)
    x, hyp = x[:n], hyp[:n]
    art, z = yasa.art_detect(x[None], sf=sf, window=WINDOW, hypno=hyp, include=INCLUDE,
                             method="std", threshold=THRESHOLD)
    art = np.asarray(art).astype(int)[:n_win]
    z = np.asarray(z, float).reshape(len(art), -1)[:, 0] if np.size(z) else np.full(n_win, np.nan)
    # stage of each window: the stage of its first sample (YASA epochs the hypnogram the same way)
    w_stage = hyp[(np.arange(n_win) * WINDOW * sf).astype(int)]
    inv = {v: k for k, v in STAGE_CODE.items()}
    start = np.arange(n_win) * WINDOW
    return pd.DataFrame(dict(start=start, end=start + WINDOW,
                             stage=[inv.get(int(s), "?") for s in w_stage],
                             zscore=np.round(z, 3), artefact=art))


def events_in_artefact(ev_onsets, ev_durations, mask_df):
    """number of reference events whose centre falls in a flagged window"""
    if not len(ev_onsets):
        return 0
    c = np.asarray(ev_onsets, float) + np.nan_to_num(np.asarray(ev_durations, float)) / 2
    i = np.clip((c // WINDOW).astype(int), 0, len(mask_df) - 1)
    return int(mask_df.artefact.to_numpy()[i].sum())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spindlebench artefacts")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--recs", default=None)
    a = ap.parse_args(argv)
    from spindlebench import provenance
    provenance.track(__file__, protocols=[VERSION])

    m = pd.read_csv(paths.CACHE / "manifest.csv")
    m = m[m.dataset.isin(a.datasets.split(","))]
    if a.recs:
        m = m[m.rec.isin(a.recs.split(","))]
    rows = []
    for _, r in m.iterrows():
        z = np.load(signal_path(r.path, paths.CACHE / "manifest.csv"))
        sf = float(z["sf"])
        ch = [str(c) for c in z["ch_names"]]
        x = z["data"][ch.index(str(r.default_channel))].astype(float)
        hyp = hypno_per_sample(r.dataset, r.rec, len(x), sf)
        df = detect(x, sf, hyp)
        df.to_csv(labels_dir() / r.dataset / f"{r.rec}__artefact.csv", index=False)
        n2 = df.stage == "N2"
        row = dict(dataset=r.dataset, rec=r.rec, excluded=r.excluded, windows=len(df),
                   flagged=df.artefact.mean(), flagged_n2=df.artefact[n2].mean() if n2.any() else np.nan,
                   n2_min=n2.sum() * WINDOW / 60, n2_min_clean=(n2 & (df.artefact == 0)).sum() * WINDOW / 60)
        for ref in ("expert1", "expert2", "expert"):
            f = labels_dir() / r.dataset / f"{r.rec}__{ref}__spindle.csv"
            if f.exists():
                ev = pd.read_csv(f)
                row[f"{ref}_spindles"] = len(ev)
                row[f"{ref}_in_artefact"] = events_in_artefact(ev.onset, ev.duration, df)
        rows.append(row)
        print(r.dataset, r.rec, f"flagged {row['flagged']:.3f}", flush=True)

    out = paths.RESULTS / "artefacts"
    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    f_per = out / "per_recording.csv"
    if f_per.exists() and a.recs is None:
        old = pd.read_csv(f_per)
        per = pd.concat([old[~old.dataset.isin(per.dataset.unique())], per], ignore_index=True)
    per.to_csv(f_per, index=False)
    print(summary(per).to_string())


def summary(per):
    g = per.groupby("dataset")
    s = pd.DataFrame({
        "recordings": g.size(),
        "flagged_%": 100 * g.flagged.mean(),
        "flagged_N2_%": 100 * g.flagged_n2.mean(),
        "flagged_N2_max_%": 100 * g.flagged_n2.max(),
        "N2_h": g.n2_min.sum() / 60,
        "N2_h_clean": g.n2_min_clean.sum() / 60,
    })
    for ref in ("expert1", "expert2", "expert"):
        if f"{ref}_spindles" in per:
            has = g[f"{ref}_spindles"].count() > 0
            txt = g[f"{ref}_in_artefact"].sum().astype(int).astype(str) + " / " + \
                g[f"{ref}_spindles"].sum().astype(int).astype(str)
            s[f"{ref}_in_artefact"] = txt.where(has, "-")
    s = s.round(2)
    s.to_csv(paths.RESULTS / "artefacts" / "summary.csv")
    return s


if __name__ == "__main__":
    main()
