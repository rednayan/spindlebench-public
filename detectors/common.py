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
    return p.parse_args()


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
        yield row, z["data"], float(z["sf"]), ch_names, chans


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
    d = Path(args.out) / row.dataset / detector
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.sort_values(["channel", "onset"]).to_csv(d / f"{row.rec}.csv", index=False)
    return len(df)
