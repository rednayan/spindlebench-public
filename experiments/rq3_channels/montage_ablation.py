"""
RQ3 montage control: is part of the DREAMS -> NAP drop the reference montage?

    uv run python experiments/rq3_channels/montage_ablation.py build    # cache_montage/ (NAP C3 against TP9 only)
    uv run python experiments/rq3_channels/montage_ablation.py detect   # every spindle detector on it
    uv run python experiments/rq1_benchmark/evaluate.py --store results/sensitivity/montage/detections \
        --datasets nap --out results/sensitivity/montage

DREAMS spindle excerpts are C3 (or Cz) against one ear (A1); the NAP benchmark uses C3 against linked
mastoids (x - TP10/2, Section 3.3). NAP was recorded against TP9, so C3 against TP9 alone is the
single-mastoid montage closest to DREAMS: in the cache TP10 holds TP10/2, hence C3_TP9 = C3 + TP10.
Labels, hypnograms and scoring windows are the main cache's; only the signal changes.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.data.cache import signal_path
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

OUT = paths.CACHE_MONTAGE
# a venv interpreter started from another venv's Python inherits its launcher variables and then
# loads the wrong standard library; give every detector environment a clean one
CLEAN_ENV = {k: v for k, v in os.environ.items() if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH")}
STORE = str(paths.RESULTS / "sensitivity" / "montage" / "detections")
RUNS = [  # (interpreter, script, extra args)
    (".venv/Scripts/python", "detectors/yasa/run.py", []),
    (".venv/Scripts/python", "detectors/a7/run.py", []),
    (".venv/Scripts/python", "detectors/moelle/run.py", []),
    ("detectors/wonambi/.venv/Scripts/python", "detectors/wonambi/run.py", []),
    (".venv/Scripts/python", "detectors/sumo/run.py", ["--model", "sumo,sumov2"]),
    ("detectors/openspindlenet/.venv/Scripts/python", "detectors/openspindlenet/run.py", []),
    ("detectors/sssm/.venv/Scripts/python", "detectors/sssm/run.py", []),
]


def build():
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    m = m[m.dataset == "nap"].copy()
    (OUT / "nap").mkdir(parents=True, exist_ok=True)
    files = []
    for row in m.itertuples():
        z = np.load(signal_path(row.path))
        ch = [str(c) for c in z["ch_names"]]
        x = z["data"][ch.index("C3")] + z["data"][ch.index("TP10")]  # undo linked mastoids: C3 - TP9
        p = OUT / "nap" / f"{row.rec}.npz"
        np.savez(p, data=x[None, :].astype(np.float32), sf=z["sf"], ch_names=np.array(["C3"]))
        files.append(p.relative_to(OUT).as_posix())  # relative to the manifest
    m["path"] = files
    m["default_channel"] = "C3"
    m.to_csv(OUT / "manifest.csv", index=False)
    print("wrote", len(m), "recordings to", OUT)


def detect():
    for py, script, extra in RUNS:
        cmd = [str(paths.ROOT / py), str(paths.ROOT / script),
               "--manifest", str(OUT / "manifest.csv"), "--out", STORE, "--datasets", "nap"] + extra
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, env=CLEAN_ENV)
        print(script, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    {"build": build, "detect": detect}[sys.argv[1]]()
