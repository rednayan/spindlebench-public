"""
Export every benchmark recording to one common format, so any detector (rule-based
or learned, in any environment) reads the same input and the evaluator reads the
same labels.

    uv run spindlebench cache                # writes the cache (default <repo>/cache, gitignored)
    uv run spindlebench cache --labels-only  # NAP: re-parse labels, keep the cached signals
    uv run spindlebench cache --boas-only    # append the BOAS subset (spindlebench download boas)

cache/manifest.csv           one row per recording: dataset, rec, npz path (relative to the
                             manifest, forward slashes), sf, default
                             channel, duration, per-annotator scoring windows, excluded flag
cache/signals/<dataset>/<rec>.npz
                             data (n_ch, n_samples) float32 in µV, sf, ch_names
cache/labels/<dataset>/<rec>__<annotator>__<event>.csv
                             onset,duration  (duration NaN when only onsets were marked)
cache/labels/<dataset>/<rec>__hypno.csv
                             epoch_start,stage  (W N1 N2 N3 REM ?)

Signal preprocessing is deliberately minimal and identical for every detector:
  DREAMS   the scored channel only, native sampling rate (50/100/200 Hz), µV
  NAP      all 64 channels, linked-mastoid reference (x - TP10/2; TP9 was the
           recording reference), 0.1-45 Hz band-pass (DC-coupled amplifier),
           resampled to 200 Hz
  BOAS     the six PSG EEG channels (F3 F4 C3 C4 O1 O2) against the left mastoid as
           recorded, native 256 Hz, no filter; staging only, no events (label-free analyses)
Each detector applies its own preprocessing on top (as its authors specify).
"""

import json
import sys

import mne
import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.data import nap_annotations as na

mne.set_log_level("ERROR")
CACHE = paths.CACHE
RKMAP = {5: "W", 4: "REM", 3: "N1", 2: "N2", 1: "N3", 0: "N3"}  # DREAMS R&K codes
manifest = []


def write_events(path, ev):
    pd.DataFrame(ev, columns=["onset", "duration"]).to_csv(path, index=False)


def write_hypno(path, stages, epoch):
    pd.DataFrame({"epoch_start": np.arange(len(stages)) * epoch, "stage": stages}).to_csv(path, index=False)


def read_dreams_events(path):
    return np.loadtxt(path, skiprows=1, ndmin=2) if path.exists() else None


def dreams(db, dataset, n_ex, event):
    sig_dir, lab_dir = CACHE / "signals" / dataset, CACHE / "labels" / dataset
    sig_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, n_ex + 1):
        rec = f"excerpt{n}"
        raw = mne.io.read_raw_edf(db / f"{rec}.edf", preload=True)
        ch = next(c for c in raw.ch_names if c.upper().startswith(("C3", "CZ")))
        x = raw.get_data(picks=ch)[0]
        if np.abs(x).max() < 1e-2:  # volts -> µV (some headers already give µV)
            x = x * 1e6
        sf = raw.info["sfreq"]
        np.savez_compressed(sig_dir / f"{rec}.npz", data=x[None].astype(np.float32), sf=sf, ch_names=np.array([ch]))
        windows = {}
        for k in (1, 2):
            ev = read_dreams_events(db / f"Visual_scoring{k}_{rec}.txt")
            if ev is None:
                continue
            write_events(lab_dir / f"{rec}__expert{k}__{event}.csv", ev)
            # spindle expert 1 stopped at ~1000 s in every excerpt (last marks 878-996 s)
            windows[f"expert{k}"] = [0.0, 1000.0] if (event == "spindle" and k == 1) else [0.0, len(x) / sf]
        auto = read_dreams_events(db / f"Automatic_detection_{rec}.txt")
        if auto is not None:  # the DREAMS authors' own detector, kept as a detector output
            d = paths.RESULTS / "detections" / dataset / "dreams_auto"
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"onset": auto[:, 0], "duration": auto[:, 1], "channel": ch, "event": event}).to_csv(
                d / f"{rec}.csv", index=False
            )
        hyp = np.loadtxt(db / f"Hypnogram_{rec}.txt", skiprows=1).astype(int)
        write_hypno(lab_dir / f"{rec}__hypno.csv", [RKMAP.get(h, "?") for h in hyp], 5.0)
        manifest.append(
            dict(
                dataset=dataset,
                rec=rec,
                subject=n,
                path=(sig_dir / f"{rec}.npz").relative_to(CACHE).as_posix(),
                sf=sf,
                default_channel=ch,
                duration=len(x) / sf,
                n_channels=1,
                events=event,
                windows=json.dumps(windows),
                excluded=False,
            )
        )
        print(dataset, rec, sf, ch, {k: v for k, v in windows.items()}, flush=True)


def nap(root, labels_only=False):
    sig_dir, lab_dir = CACHE / "signals" / "nap", CACHE / "labels" / "nap"
    sig_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for r in na.recordings(root):
        if r["ann"] is None or r["vhdr"] is None or not r["vhdr"].with_suffix(".eeg").exists():
            continue
        rec = f"suj{r['subject']}_d{r['day']}"
        a = na.parse_annotations(r["ann"])
        if labels_only:
            with np.load(sig_dir / f"{rec}.npz") as z:
                n_ch, n_samp = z["data"].shape
        else:
            raw = mne.io.read_raw_brainvision(r["vhdr"], preload=True)
            raw.drop_channels([c for c in ["Aux1"] if c in raw.ch_names])
            raw.set_channel_types({c: "eog" for c in ["LOc", "ROc"] if c in raw.ch_names})
            x = raw.get_data(picks="all")
            x = x - x[raw.ch_names.index("TP10")] / 2  # linked mastoids
            raw._data = x
            raw.filter(0.1, 45.0, picks="all")
            raw.resample(200.0)
            data = (raw.get_data(picks="all") * 1e6).astype(np.float32)
            np.savez_compressed(sig_dir / f"{rec}.npz", data=data, sf=200.0, ch_names=np.array(raw.ch_names))
            n_ch, n_samp = data.shape
        dur = n_samp / 200.0
        sp = a["spindles"][a["spindles"] < dur]
        kc = a["kcomplexes"][a["kcomplexes"] < dur]
        write_events(lab_dir / f"{rec}__expert__spindle.csv", np.c_[sp, np.full(len(sp), np.nan)])
        write_events(lab_dir / f"{rec}__expert__kcomplex.csv", np.c_[kc, np.full(len(kc), np.nan)])
        hyp = na.hypnogram(a["stages"], dur, a["stage_offs"])
        write_hypno(lab_dir / f"{rec}__hypno.csv", list(hyp), 30.0)
        # the rater scored up to the last staged epoch; the recording runs on past it
        scored = np.flatnonzero(hyp != "?")
        t1 = min(dur, (scored[-1] + 1) * na.EPOCH) if len(scored) else dur
        manifest.append(
            dict(
                dataset="nap",
                rec=rec,
                subject=r["subject"],
                path=(sig_dir / f"{rec}.npz").relative_to(CACHE).as_posix(),
                sf=200.0,
                default_channel="C3",
                duration=dur,
                n_channels=n_ch,
                events="spindle,kcomplex",
                windows=json.dumps({"expert": [0.0, t1]}),
                excluded=(r["subject"], r["day"]) in na.EXCLUDED,
            )
        )
        print("nap", rec, (n_ch, n_samp), f"scored to {t1:.0f} s", f"{len(sp)} sp, {len(kc)} kc", flush=True)


BOAS_STAGES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}  # 8 disconnection, -2 artefact -> "?"


def boas(root):
    """BOAS PSG nights: F3 F4 C3 C4 O1 O2 against M1, native 256 Hz, µV; consensus staging; no events"""
    sig_dir, lab_dir = CACHE / "signals" / "boas", CACHE / "labels" / "boas"
    sig_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for sub in sorted(p.name for p in root.glob("sub-*")):
        f = root / sub / f"{sub}_task-Sleep_acq-psg_eeg.edf"
        raw = mne.io.read_raw_edf(f, preload=True)
        want = {
            c: c.removeprefix("PSG_")
            for c in raw.ch_names
            if c.removeprefix("PSG_") in ("F3", "F4", "C3", "C4", "O1", "O2")
        }
        data = (raw.get_data(picks=list(want)) * 1e6).astype(np.float32)
        sf = raw.info["sfreq"]
        rec = sub.replace("-", "")
        np.savez_compressed(sig_dir / f"{rec}.npz", data=data, sf=sf, ch_names=np.array(list(want.values())))
        ev = pd.read_csv(root / sub / f"{sub}_task-Sleep_acq-psg_events.tsv", sep="\t")
        write_hypno(lab_dir / f"{rec}__hypno.csv", [BOAS_STAGES.get(int(s), "?") for s in ev.stage_hum], 30.0)
        manifest.append(
            dict(
                dataset="boas",
                rec=rec,
                subject=int(sub[4:]),
                path=(sig_dir / f"{rec}.npz").relative_to(CACHE).as_posix(),
                sf=sf,
                default_channel="C3",
                duration=data.shape[1] / sf,
                n_channels=len(want),
                events="",
                windows=json.dumps({}),
                excluded=False,
            )
        )
        print("boas", rec, data.shape, sf, flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--boas-only" in argv:  # append BOAS (spindlebench download boas) to an existing cache
        boas(paths.raw("boas"))
        old = pd.read_csv(CACHE / "manifest.csv")
        pd.concat([old[old.dataset != "boas"], pd.DataFrame(manifest)]).to_csv(CACHE / "manifest.csv", index=False)
        print(f"appended {len(manifest)} BOAS recordings to {CACHE}/manifest.csv")
        return
    labels_only = "--labels-only" in argv
    old = pd.read_csv(CACHE / "manifest.csv") if (CACHE / "manifest.csv").exists() else None
    dreams(paths.raw("dreams_spindles"), "dreams_spindles", 8, "spindle")
    dreams(paths.raw("dreams_kc"), "dreams_kc", 10, "kcomplex")
    nap(paths.raw("nap"), labels_only=labels_only)
    rows = pd.DataFrame(manifest)
    if labels_only and old is not None:  # BOAS has no labels to re-parse: keep its rows
        rows = pd.concat([rows, old[old.dataset == "boas"]])
    elif paths.raw("boas").exists():
        n = len(manifest)
        boas(paths.raw("boas"))
        rows = pd.DataFrame(manifest)
        print(f"BOAS: {len(manifest) - n} recordings")
    rows.to_csv(CACHE / "manifest.csv", index=False)
    manifest[:] = rows.to_dict("records")
    print(f"wrote {len(manifest)} recordings to {CACHE}/manifest.csv")


if __name__ == "__main__":
    main()
