"""
RQ1 "usable in practice": wall time of each detector adapter on one NAP nap (C3, ~30 min), measured
the same way for every detector: the whole adapter process, imports and model loading included,
into a scratch store. Run on an otherwise idle machine.

    uv run python experiments/usability/usability_timing.py [--rec suj11_d1] [--repeats 3]

Writes results/usability/runtime.csv (median over repeats). The usability table itself is
docs/thesis/detector_usability.md.
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from spindlebench import paths
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

RUNS = {  # adapter -> (interpreter, script, extra args); detectors written by one run share its time
    "yasa (3 configs)": (".venv/Scripts/python", "detectors/yasa/run.py", []),
    "a7 (4 configs)": (".venv/Scripts/python", "detectors/a7/run.py", []),
    "moelle2011_faithful (2)": (".venv/Scripts/python", "detectors/moelle/run.py", []),
    "wonambi (11 methods)": ("detectors/wonambi/.venv/Scripts/python", "detectors/wonambi/run.py", []),
    "sumo": (".venv/Scripts/python", "detectors/sumo/run.py", ["--model", "sumo"]),
    "sumov2": (".venv/Scripts/python", "detectors/sumo/run.py", ["--model", "sumov2"]),
    "openspindlenet": ("detectors/openspindlenet/.venv/Scripts/python", "detectors/openspindlenet/run.py", []),
    "sssm (3 configs, spindles + K-complexes)": ("detectors/sssm/.venv/Scripts/python", "detectors/sssm/run.py", []),
    "spinky_kc": (".venv/Scripts/python", "detectors/kcomplex_rules/run.py", ["--detectors", "spinky_kc"]),
    "yasa_sw_kc": (".venv/Scripts/python", "detectors/kcomplex_rules/run.py", ["--detectors", "yasa_sw_kc"]),
}
SCRATCH = paths.RESULTS / "usability/_detections"
# see experiments/rq3_channels/montage_ablation.py: child venv interpreters must not inherit this process's launcher variables
CLEAN_ENV = {k: v for k, v in os.environ.items() if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="suj11_d1")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    rows = []
    for name, (py, script, extra) in RUNS.items():
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            r = subprocess.run([str(Path(py).resolve()), script, "--datasets", "nap", "--recs", a.rec,
                                "--out", str(SCRATCH)] + extra, capture_output=True, text=True, env=CLEAN_ENV)
            ts.append(time.perf_counter() - t0)
            if r.returncode:
                print(name, "failed:", r.stderr[-300:])
                break
        rows.append(dict(adapter=name, rec=a.rec, seconds_median=sorted(ts)[len(ts) // 2], repeats=len(ts)))
        print(rows[-1], flush=True)
    out = paths.RESULTS / "usability"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "runtime.csv", index=False)
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
