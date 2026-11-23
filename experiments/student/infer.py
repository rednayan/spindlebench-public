"""
The one gold-set run (plan 6.4, 8): every frozen student on DREAMS and NAP.

    uv run python infer.py [--runs a,b] [--track cpu|gpu] [--device auto|cpu|cuda]
                                              (from experiments/student; after selection.py freeze + commit;
                                               --track gpu reads frozen_gpu.json; the network runs on --device,
                                               auto = the GPU when torch sees one)

This is the only script in the study that opens DREAMS or NAP, and it reads their signals only: no expert
label, no expert hypnogram. It refuses to run unless
  1. results/student/selection/frozen.json exists, is tracked by git and has no uncommitted change,
  2. the working tree has no uncommitted change to tracked files,
  3. every checkpoint's sha256 equals the one frozen.
Per frozen run (not excluded) and gold recording: the default channel (DREAMS as listed, NAP C3), SUMOv2
preprocessing, whole-recording inference (validate.predict_night), the frozen decoder setting ->
  results/student/detections/<dataset>/<run>/<rec>.csv   detection file contract (detector id = run name)
  results/student/work/gold_p/<run>/<dataset>/<rec>.npz  p(t), b(t) at 100 Hz, float16 (calibration, step 5)
Then score with experiments/rq1_benchmark/evaluate.py --store results/student/detections (README).
"""
import argparse
import json
import subprocess

import numpy as np
import pandas as pd
import torch

import data
import decode
import selection
import validate
from spindlebench import paths, provenance

FROZEN = selection.SEL / "frozen.json"
OUT = data.STUDY / "detections"
GOLD_P = data.WORK / "gold_p"


class NotFrozen(RuntimeError):
    pass


def git(*a):
    return subprocess.run(["git", "-C", str(paths.ROOT), *a], capture_output=True, text=True)


def check_frozen(frozen=FROZEN):
    """-> the frozen dict, or raise NotFrozen"""
    if not frozen.exists():
        raise NotFrozen(f"{frozen} does not exist: run selection.py freeze and commit it first")
    rel = frozen.relative_to(paths.ROOT).as_posix()
    if git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise NotFrozen(f"{rel} is not committed")
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise NotFrozen(f"{rel} has uncommitted changes")
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise NotFrozen("the working tree has uncommitted changes to tracked files")
    fz = json.loads(frozen.read_text())
    for name, r in fz["runs"].items():
        if r.get("excluded"):
            continue
        ck = data.RUNS / r["checkpoint"]
        if selection.sha256(ck) != r["sha256"]:
            raise NotFrozen(f"checkpoint of {name} changed since the freeze")
    return fz


def events(p, b, s):
    if s["decoder"] == "D0":
        ev = decode.d0(p, s["tau"])
    elif s["decoder"] == "D1":
        ev = decode.d1(p, s["step"], s["gamma"], s["merge"], s["lam"])
    else:
        ev = decode.d2(p)
    return decode.snap(ev, b) if s.get("snap") and b is not None else ev


def frozen_file(track):
    """the freeze of a track, as selection.py freeze writes it"""
    return FROZEN if track == "cpu" else selection.SEL / f"frozen_{track}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--track", default="cpu", help="the track whose freeze is run (cpu: frozen.json, else "
                                                   "frozen_<track>.json)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="where the network runs; auto = cuda when torch sees a GPU")
    a = ap.parse_args()
    fz = check_frozen(frozen_file(a.track))
    if fz.get("track", "cpu") != a.track:
        raise NotFrozen(f"{frozen_file(a.track).name} is the freeze of track {fz.get('track')}, not {a.track}")
    provenance.track(__file__, protocols=["benchmark.v1", "nap_points.v1", "artefacts.v1"])
    device = selection.resolve_device(a.device)
    selection.setup_device(device)
    torch.set_num_threads(4)
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    m = m[m.dataset.isin(data.GOLD)]
    runs = [r for r, v in fz["runs"].items() if not v.get("excluded")]
    if a.runs:
        runs = [r for r in runs if r in a.runs.split(",")]
    print(f"{len(runs)} frozen runs of track {a.track}, {len(m)} gold recordings, network on {device}", flush=True)
    for name in runs:
        net, cfg, done, ck = selection.load_run(name, device)
        s = fz["runs"][name]["setting"]
        for row in m.itertuples():
            x, sf = data.load_signal(row)
            y = data.preprocess(x, sf)
            p, b = validate.predict_night(net, y)
            b = b if net.boundary is not None else None
            ev = events(p, b, s)
            rows = [dict(onset=round(float(on), 3), duration=round(float(off - on), 3), channel=str(row.default_channel),
                         event="spindle", score=round(float(sc), 4)) for on, off, sc in ev]
            f = OUT / row.dataset / name / f"{row.rec}.csv"
            f.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=["onset", "duration", "channel", "event", "score"]).to_csv(f, index=False)
            g = GOLD_P / name / row.dataset / f"{row.rec}.npz"
            g.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(g, p=p.astype(np.float16), b=(b if b is not None else np.zeros(0)).astype(np.float16))
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
