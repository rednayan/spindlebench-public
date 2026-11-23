"""
RQ3 preprocessing control: score NAP after the dataset authors' own preprocessing instead of ours.

    uv run python experiments/rq3_channels/authors_pipeline.py build    # cache_authors/ (C3 after the authors' steps)
    uv run python experiments/rq3_channels/authors_pipeline.py detect   # the five deck detectors on it
    uv run python experiments/rq1_benchmark/evaluate.py --store results/sensitivity/authors_pipeline/detections \
        --datasets nap --out results/sensitivity/authors_pipeline

Our cache (export_cache.nap): linked mastoids (x - TP10/2), 0.1-45 Hz, 200 Hz. The authors (Mei et al. 2017,
PMC5650544; Mei et al. 2018, PMC5998176) downsampled to 500 Hz, re-referenced to the common average, filtered
0.1-200 Hz with notches at 60/120/180 Hz, removed EOG-correlated ICA components automatically, then filtered
0.1-50 Hz with zero-phase FIR filters (MNE). This script follows that order. The papers do not give the ICA
algorithm, the number of components or the EOG threshold; here: FastICA, 20 components, fitted on a 1 Hz
high-passed copy (MNE's recommendation), components found with find_bads_eog on LOc and ROc (default threshold).
Labels, hypnograms and scoring windows are the main cache's; only the signal changes.
"""
import os
import subprocess
import sys

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from spindlebench import paths
from spindlebench import provenance
from spindlebench.data import nap_annotations as na

mne.set_log_level("ERROR")
OUT = paths.CACHE.parent / "cache_authors"
STORE = str(paths.RESULTS / "sensitivity" / "authors_pipeline" / "detections")
CLEAN_ENV = {k: v for k, v in os.environ.items() if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH")}
RUNS = [  # the five detectors of the progress-update deck
    (".venv/Scripts/python", "detectors/yasa/run.py", []),
    ("detectors/wonambi/.venv/Scripts/python", "detectors/wonambi/run.py", ["--methods", "Moelle2011"]),
    (".venv/Scripts/python", "detectors/sumo/run.py", ["--model", "sumo,sumov2"]),
    ("detectors/openspindlenet/.venv/Scripts/python", "detectors/openspindlenet/run.py", []),
]
SF = 500.0


def one(vhdr, rec):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    raw.drop_channels([c for c in ["Aux1"] if c in raw.ch_names])
    raw.set_channel_types({c: "eog" for c in ["LOc", "ROc"] if c in raw.ch_names})
    raw.resample(SF)
    raw.set_eeg_reference("average", projection=False)
    raw.filter(0.1, 200.0, picks="all")
    raw.notch_filter([60.0, 120.0, 180.0], picks="all")
    fit = raw.copy().filter(1.0, None, picks="all")
    ica = mne.preprocessing.ICA(n_components=20, method="fastica", random_state=97, max_iter="auto")
    ica.fit(fit, picks="eeg", decim=5)
    bad, _ = ica.find_bads_eog(raw, ch_name=["LOc", "ROc"])
    ica.exclude = sorted(set(bad))
    ica.apply(raw)
    raw.filter(0.1, 50.0, picks="all", method="fir", phase="zero")
    x = raw.get_data(picks=["C3"])[0] * 1e6
    p = OUT / "nap" / f"{rec}.npz"
    np.savez(p, data=x[None, :].astype(np.float32), sf=SF, ch_names=np.array(["C3"]))
    return rec, len(ica.exclude)


def build():
    provenance.track(__file__)
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    m = m[m.dataset == "nap"].copy()
    (OUT / "nap").mkdir(parents=True, exist_ok=True)
    raws = {f"suj{r['subject']}_d{r['day']}": r["vhdr"] for r in na.recordings(paths.DATA / "NapEEG") if r["vhdr"] is not None}
    todo = [(raws[rec], rec) for rec in m.rec if rec in raws]
    res = Parallel(n_jobs=5, verbose=5)(delayed(one)(v, r) for v, r in todo)
    for rec, n in res:
        print(rec, "ICA components removed:", n, flush=True)
    done = {r for r, _ in res}
    m = m[m.rec.isin(done)].copy()
    m["path"] = [f"nap/{r}.npz" for r in m.rec]
    m["sf"], m["default_channel"], m["n_channels"] = SF, "C3", 1
    m.to_csv(OUT / "manifest.csv", index=False)
    pd.DataFrame(res, columns=["rec", "ica_removed"]).to_csv(OUT / "ica_removed.csv", index=False)
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
