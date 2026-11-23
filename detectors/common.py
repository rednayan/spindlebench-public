"""
Shared helpers for detector adapters: iterate the cache, write detection files.
Import from an adapter with:  sys.path.insert(0, str(Path(__file__).parents[1])); import common
Only numpy + pandas are needed, so it works inside any detector environment (Python 3.9+).

The first write of each (dataset, detector) in a process also writes a provenance record to
results/provenance/detections/<dataset>/<detector>.json (same fields as spindlebench.provenance).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["onset", "duration", "channel", "event", "score"]
NAP_CHANNELS = ["C3", "F3", "F4", "C4", "O1", "O2", "Cz"]
# same defaults and environment overrides as spindlebench.paths (this file cannot import the package)
REPO = Path(os.environ.get("SPINDLEBENCH_ROOT") or Path(__file__).resolve().parents[1])
CACHE = Path(os.environ.get("SPINDLEBENCH_CACHE") or REPO / "cache")
RESULTS = Path(os.environ.get("SPINDLEBENCH_RESULTS") or REPO / "results")


def parse_args(default_datasets):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(CACHE / "manifest.csv"))
    p.add_argument("--out", default=str(RESULTS / "detections"))
    p.add_argument("--datasets", default=",".join(default_datasets))
    p.add_argument("--channels", default=None, help="NAP channels, default C3")
    p.add_argument("--recs", default=None)
    p.add_argument("--clean", action="store_true",
                   help="run on artefact-free windows only (cache/labels/<ds>/<rec>__artefact.csv, artefacts.v1); "
                        "default output results/detections_clean")
    a = p.parse_args()
    if a.clean and a.out == str(RESULTS / "detections"):
        a.out = str(RESULTS / "detections_clean")
    return a


# --clean: the detector sees the clean windows joined end to end (as Luna does with masked epochs);
# its events are mapped back to recording time. Events that cross a join, or lie within JOIN_GUARD s
# of one (filter transients at the step), are dropped.
JOIN_GUARD = 1.0
_CLEAN = {}


def clean_view(row, data, sf):
    """(data restricted to clean samples, original sample index of each kept sample)"""
    a = pd.read_csv(CACHE / "labels" / row.dataset / f"{row.rec}__artefact.csv")
    keep = np.ones(data.shape[1], bool)
    for s, e in a.loc[a.artefact == 1, ["start", "end"]].to_numpy():
        keep[int(round(s * sf)):int(round(e * sf))] = False
    idx = np.flatnonzero(keep)
    return data[:, idx], idx


def to_recording_time(rows, idx, sf):
    """map events from joined-clean time back to recording time; drop those at or near a join"""
    if not len(idx):
        return []
    joins = np.flatnonzero(np.diff(idx) != 1) + 1  # first clean-sample index after each join
    out = []
    for r in rows:
        i0 = int(round(r["onset"] * sf))
        i1 = int(round((r["onset"] + r["duration"]) * sf))
        i0, i1 = max(i0, 0), min(max(i1, i0), len(idx) - 1)
        if len(joins):
            g = int(JOIN_GUARD * sf)
            k = np.searchsorted(joins, i0 - g)
            if k < len(joins) and joins[k] <= i1 + g:
                continue
        o = dict(r)
        o["onset"] = idx[i0] / sf
        o["duration"] = (idx[i1] - idx[i0]) / sf
        out.append(o)
    return out


def signal_path(path, manifest_path):
    """a manifest's `path` entry: relative to the manifest's folder (current format) or to the
    repository root (older caches); mirrors spindlebench.data.cache.signal_path"""
    p = Path(str(path).replace("\\", "/"))
    if p.is_absolute():
        return p
    q = Path(manifest_path).parent / p
    return q if q.exists() else REPO / p


def iter_recordings(args):
    """yield (row, data[n_ch, n], sf, ch_names, channels_to_run)"""
    m = pd.read_csv(args.manifest)
    m = m[m.dataset.isin(args.datasets.split(","))]
    if args.recs:
        m = m[m.rec.isin(args.recs.split(","))]
    for _, row in m.iterrows():
        z = np.load(signal_path(row.path, args.manifest))
        ch_names = [str(c) for c in z["ch_names"]]
        if row.dataset == "nap":
            chans = args.channels.split(",") if args.channels else [row.default_channel]
        else:
            chans = [row.default_channel]
        data, sf = z["data"], float(z["sf"])
        if getattr(args, "clean", False):
            # only the channels that will be run are kept (memory: CinC nights have six)
            rows_ = [ch_names.index(c) for c in chans]
            data, ch_names = data[rows_], [ch_names[i] for i in rows_]
            data, idx = clean_view(row, data, sf)
            _CLEAN[(row.dataset, row.rec)] = (idx, sf)
        yield row, data, sf, ch_names, chans


PACKAGES = ["numpy", "pandas", "scipy", "mne", "yasa", "torch", "onnxruntime", "wonambi", "sssm", "openspindlenet",
            "pytorch-lightning", "lightning"]
_recorded = set()


def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), *a], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def scrub(obj, root):
    """paths under the repository become <root>/..., the home folder ~/...: records are published"""
    if isinstance(obj, Path):
        obj = str(obj)
    if isinstance(obj, dict):
        return {k: scrub(v, root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v, root) for v in obj]
    if isinstance(obj, str):
        s = obj.replace("\\", "/")
        for base, name in ((Path(root).resolve().as_posix(), "<root>"), (Path.home().as_posix(), "~")):
            if s.lower().startswith(base.lower()):
                return name + s[len(base):]
        return obj
    return obj


def write_provenance(args, dataset, detector):
    """one record per (dataset, detector) run; mirrors spindlebench.provenance.record"""
    versions = {}
    for n in PACKAGES:
        try:
            versions[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    commit = _git("rev-parse", "HEAD")
    rec = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": [sys.executable, *sys.argv],
           "git": {"commit": commit or None, "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
                   "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")) if commit else None},
           "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
           "config": {k: v for k, v in vars(args).items()}}
    rec = scrub(rec, REPO)
    d = Path(args.out).parent / "provenance" / "detections" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{detector}.json").write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")


def write(args, row, detector, rows):
    """rows: list of dicts with onset, duration, channel, event[, score]"""
    if (row.dataset, detector) not in _recorded:
        _recorded.add((row.dataset, detector))
        write_provenance(args, row.dataset, detector)
    if getattr(args, "clean", False):
        idx, sf = _CLEAN[(row.dataset, row.rec)]
        rows = to_recording_time(rows, idx, sf)
    d = Path(args.out) / row.dataset / detector
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.sort_values(["channel", "onset"]).to_csv(d / f"{row.rec}.csv", index=False)
    return len(df)
