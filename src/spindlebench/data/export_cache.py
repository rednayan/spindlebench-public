"""
Export every benchmark recording to one common format, so any detector (rule-based
or learned, in any environment) reads the same input and the evaluator reads the
same labels.

    uv run spindlebench cache                # writes the cache (default <repo>/cache, gitignored)
    uv run spindlebench cache --labels-only  # NAP: re-parse labels, keep the cached signals
    uv run spindlebench cache --boas-only    # append the BOAS subset (spindlebench download boas)
    uv run spindlebench cache --only hmc,cinc2018   # append (or refresh) the named label-free datasets

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
  HMC      the four EEG channels (F4-M1 C4-M1 O2-M1 C3-M2) as recorded, native 256 Hz, no
           filter; AASM staging only, no events
  CinC2018 the six EEG channels (F3-M2 F4-M1 C3-M2 C4-M1 O1-M2 O2-M1) as released (200 Hz),
           read with wfdb; AASM staging from the .arousal file, no events
Each detector applies its own preprocessing on top (as its authors specify).
"""

import json
import sys

import mne
import wfdb
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


def label_free_row(dataset, rec, subject, path, sf, default_channel, data):
    return dict(dataset=dataset, rec=rec, subject=subject, path=path.relative_to(CACHE).as_posix(), sf=sf,
                default_channel=default_channel, duration=data.shape[1] / sf, n_channels=data.shape[0],
                events="", windows=json.dumps({}), excluded=False)


HMC_STAGES = {"W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "R": "REM"}


def hmc(root):
    """HMC nights: the EEG channels as recorded (mastoid references), native 256 Hz, µV; AASM staging
    from <rec>_sleepscoring.txt (30-s epochs, onset in seconds from the recording start); no events"""
    sig_dir, lab_dir = CACHE / "signals" / "hmc", CACHE / "labels" / "hmc"
    sig_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for rec in (root / "RECORDS").read_text().split():
        rec = rec.split("/")[-1].removesuffix(".edf")  # RECORDS lists recordings/SNxxx.edf
        raw = mne.io.read_raw_edf(root / "recordings" / f"{rec}.edf", preload=True)
        eeg = [c for c in raw.ch_names if c.startswith("EEG ")]
        data = (raw.get_data(picks=eeg) * 1e6).astype(np.float32)
        sf = raw.info["sfreq"]
        names = [c.removeprefix("EEG ") for c in eeg]
        np.savez_compressed(sig_dir / f"{rec}.npz", data=data, sf=sf, ch_names=np.array(names))
        sc = pd.read_csv(root / "recordings" / f"{rec}_sleepscoring.txt", skipinitialspace=True)
        sc = sc[sc["Annotation"].str.startswith("Sleep stage ")]
        stages = np.full(int(np.ceil(data.shape[1] / sf / 30.0)), "?", dtype=object)
        for onset, ann in zip(sc["Recording onset"].astype(float), sc["Annotation"]):
            i = int(round(onset / 30.0))
            assert abs(onset - 30.0 * i) < 1e-3, (rec, onset)
            if i < len(stages):
                stages[i] = HMC_STAGES[ann.removeprefix("Sleep stage ")]
        write_hypno(lab_dir / f"{rec}__hypno.csv", list(stages), 30.0)
        default = "C3-M2" if "C3-M2" in names else "C4-M1"
        manifest.append(label_free_row("hmc", rec, int(rec[2:]), sig_dir / f"{rec}.npz", sf, default, data))
        print("hmc", rec, data.shape, sf, flush=True)


CINC_EEG = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]
CINC_STAGES = {"W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "R": "REM"}


def cinc2018(root):
    """CinC 2018 training subset (spindlebench download cinc2018): six EEG channels against the
    contralateral mastoid as released, 200 Hz, µV. Stages are change points in the .arousal
    annotation file (aux_note W N1 N2 N3 R); each lasts until the next one, and epochs before the
    first are unscored ("?"). No events."""
    sig_dir, lab_dir = CACHE / "signals" / "cinc2018", CACHE / "labels" / "cinc2018"
    sig_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for rec in pd.read_csv(root / "records_used.csv").Record:
        base = str(root / "training" / rec / rec)
        r = wfdb.rdrecord(base, channel_names=CINC_EEG)
        assert r.sig_name == CINC_EEG and set(r.units) == {"uV"}, (rec, r.sig_name, r.units)
        data = r.p_signal.T.astype(np.float32)
        sf = float(r.fs)
        np.savez_compressed(sig_dir / f"{rec}.npz", data=data, sf=sf, ch_names=np.array(CINC_EEG))
        ann = wfdb.rdann(base, "arousal")
        change = [(s, CINC_STAGES[n]) for s, n in zip(ann.sample, ann.aux_note) if n in CINC_STAGES]
        epoch = int(30 * sf)
        assert all(s % epoch == 0 for s, _ in change), rec
        stages = np.full(int(np.ceil(data.shape[1] / epoch)), "?", dtype=object)
        for (s, st), nxt in zip(change, [c[0] for c in change[1:]] + [len(stages) * epoch]):
            stages[s // epoch:nxt // epoch] = st
        write_hypno(lab_dir / f"{rec}__hypno.csv", list(stages), 30.0)
        subject = int(rec[2:].replace("-", ""))  # trXX-YYYY -> XXYYYY; one night per person
        manifest.append(label_free_row("cinc2018", rec, subject, sig_dir / f"{rec}.npz", sf, "C3-M2", data))
        print("cinc2018", rec, data.shape, sf, flush=True)


LABEL_FREE = {"boas": boas, "hmc": hmc, "cinc2018": cinc2018}


def append(names):
    """build the named label-free datasets and replace their rows in an existing manifest"""
    for name in names:
        LABEL_FREE[name](paths.raw(name))
    old = pd.read_csv(CACHE / "manifest.csv")
    pd.concat([old[~old.dataset.isin(names)], pd.DataFrame(manifest)]).to_csv(CACHE / "manifest.csv", index=False)
    print(f"appended {len(manifest)} recordings ({', '.join(names)}) to {CACHE}/manifest.csv")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--boas-only" in argv:  # append BOAS (spindlebench download boas) to an existing cache
        append(["boas"])
        return
    if "--only" in argv:
        append(argv[argv.index("--only") + 1].split(","))
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
    if old is not None:  # HMC and CinC 2018 are built only with --only: keep their rows
        rows = pd.concat([rows, old[old.dataset.isin(["hmc", "cinc2018"])]])
    rows.to_csv(CACHE / "manifest.csv", index=False)
    manifest[:] = rows.to_dict("records")
    print(f"wrote {len(manifest)} recordings to {CACHE}/manifest.csv")


if __name__ == "__main__":
    main()
