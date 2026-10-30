"""
The detection file contract shared by every detector adapter (detectors/INTERFACE.md).

    results/detections/<dataset>/<detector id>/<rec>.csv
    onset,duration,channel,event,score

Detector adapters run in their own environments and write these files with detectors/common.py
(numpy + pandas only). Everything downstream reads them through this module.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spindlebench import paths

COLUMNS = ["onset", "duration", "channel", "event", "score"]
EVENTS = {"spindle", "kcomplex"}
DETECTORS_DIR = paths.ROOT / "detectors"
MANIFEST_KEYS = {"name", "ids", "events", "kind", "method", "source", "adapter_licence", "environment", "command"}


def store(dataset, results=None):
    return (results or paths.RESULTS) / "detections" / dataset


def read(path):
    return pd.read_csv(path, dtype={"channel": str, "event": str})


def problems(df, duration=None, channels=None):
    """contract violations of one detection file (empty list = valid)"""
    out = []
    if list(df.columns) != COLUMNS:
        return [f"columns {list(df.columns)} != {COLUMNS}"]
    if not len(df):
        return out
    on, du = df.onset.to_numpy(float), df.duration.to_numpy(float)
    if np.isnan(on).any() or (on < 0).any():
        out.append("onset missing or negative")
    if (np.nan_to_num(du) < 0).any():
        out.append("negative duration")
    if not set(df.event) <= EVENTS:
        out.append(f"unknown events {set(df.event) - EVENTS}")
    if duration is not None and (on > duration + 1e-6).any():
        out.append(f"onset after the end of the recording ({on.max():.1f} > {duration:.1f} s)")
    if channels is not None and not set(df.channel) <= set(channels):
        out.append(f"unknown channels {set(df.channel) - set(channels)}")
    return out


def manifests():
    """detector name -> manifest dict, from detectors/<name>/manifest.json"""
    return {p.parent.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(DETECTORS_DIR.glob("*/manifest.json"))}


def owner(detector_id):
    """which adapter writes this detector id (None if unknown)"""
    for name, m in manifests().items():
        if detector_id in m["ids"]:
            return name
    return None


def id_info(detector_id):
    m = manifests().get(owner(detector_id))
    return None if m is None else {**m, "id": detector_id, "description": m["ids"][detector_id]}


def manifest_path(name):
    return Path(DETECTORS_DIR / name / "manifest.json")
