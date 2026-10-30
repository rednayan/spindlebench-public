"""
NAP EEG annotation parsing (Mei, Grossberg, Ng, Navarro & Ellmore, OSF chav7).

One expert per recording (subjects 11+): sleep stages in 30-s epochs
(Markon/Markoff), spindles and K-complexes as single time points. The marks are not
onsets: spindle marks sit near the sigma peak, K-complex marks about 0.4 s after the
trough (results/nap/annotation_alignment.png). Labels are hand-typed and messy
("Marker:Markon:2", "Makon: 2", "spind;e", "K-Complex").
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = 30.0
# excluded following Mei et al. 2017 (Methods list, restricted to the OSF naps: no N2 or too noisy):
# subjects 26, 27; subject 12 day 1; subject 20 day 2. Their Results section gives a different list.
EXCLUDED = {(26, 1), (26, 2), (27, 1), (27, 2), (12, 1), (20, 2)}
STAGE_MAP = {"w": "W", "wake": "W", "1": "N1", "2": "N2", "3": "N3", "sws": "N3", "r": "REM", "rem": "REM"}


def parse_annotations(path):
    """-> dict(stages=[(t, stage)], stage_offs=[(t, stage)], spindles=marks, kcomplexes=marks, unparsed=[labels])"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    stages, offs, sp, kc, junk = [], [], [], [], []
    for onset, label in zip(df["Onset"], df["Annotation"]):
        t = float(onset)
        s = label.strip().lower()
        if not s or s.startswith("trigger") or s == "annotation":
            continue
        if "complex" in s:
            kc.append(t)
        elif s.startswith("sp"):
            sp.append(t)
        elif re.search(r"ma\w*off", s):
            tok = s.split(":")[-1].strip()
            if tok in STAGE_MAP:
                offs.append((t, STAGE_MAP[tok]))
        elif re.search(r"ma\w*on", s):
            tok = s.split(":")[-1].strip()
            if tok in STAGE_MAP:
                stages.append((t, STAGE_MAP[tok]))
            else:
                junk.append(label)
        else:
            junk.append(label)
    return dict(stages=stages, stage_offs=offs, spindles=np.sort(sp), kcomplexes=np.sort(kc), unparsed=junk)


def hypnogram(stages, dur, offs=()):
    """30-s epoch labels. A Markon labels every epoch up to the next Markoff (or Markon);
    a Markoff with no open Markon labels the epoch it closes; other epochs stay '?'"""
    hyp = np.array(["?"] * int(np.ceil(dur / EPOCH)), dtype=object)
    # stable sort on time: of two Markons at the same instant the later one in the file wins
    marks = sorted([(t, "on", st) for t, st in stages] + [(t, "off", st) for t, st in offs], key=lambda m: m[0])
    open_ = None
    for t, kind, st in marks + [(np.inf, "on", None)]:
        if kind == "on" or open_ is not None:
            if open_ is not None:
                t0, st0 = open_
                i0, i1 = int(round(t0 / EPOCH)), max(int(round(min(t, dur) / EPOCH)), int(round(t0 / EPOCH)) + 1)
                hyp[i0:min(i1, len(hyp))] = st0
            open_ = (t, st) if kind == "on" else None
        else:  # Markoff without a Markon: label the epoch it closes
            i = int(round(t / EPOCH)) - 1
            if 0 <= i < len(hyp) and hyp[i] == "?":
                hyp[i] = st
    return hyp


def recordings(root):
    """annotation files paired with raw files on (subject, day)"""
    root = Path(root)
    ann = {}
    for p in (root / "annotations").glob("*.txt"):
        m = re.match(r"suj(\d+)_(?:nap_day|d)(\d)", p.name, re.I)
        if m:
            ann[(int(m[1]), int(m[2]))] = p
    raw = {}
    for p in (root / "raw").glob("*.vhdr"):
        m = re.match(r"suj(\d+)_l(\d)nap_day(\d)", p.name, re.I)
        if m:
            raw[(int(m[1]), int(m[3]))] = (p, int(m[2]))
    keys = sorted(set(ann) | set(raw))
    return [dict(subject=s, day=d, ann=ann.get((s, d)), vhdr=raw.get((s, d), (None, None))[0],
                 load=raw.get((s, d), (None, None))[1]) for s, d in keys]
