"""
SUMO (Kaulen et al. 2022) and SUMOv2 (Grieger et al. 2026) spindle detectors, run with the
authors' released weights (src/<repo>/output/final.ckpt) and their documented inference
preprocessing. No retraining.

    python detectors/sumo/run.py [--model sumo,sumov2,sumov2_n2] [--datasets ...] [--channels C3,F3,...]

Any interpreter can launch it: each model is re-executed inside its own environment
(.venv-sumo: Python 3.9 / torch 1.10 / Lightning 1.5.4;  .venv-sumov2: Python 3.10 / torch 2.5).

Detector ids written:
  sumo       src/sumo/scripts/predict_plain_data.py: 10th-order Butterworth high-pass 0.3 Hz +
             low-pass 30 Hz (forward-backward), resample to 100 Hz, z-score over the whole
             recording, U-Net, 0.42 s moving average, softmax argmax. No further post-processing.
  sumov2     src/sumov2/scripts/predict_edf_file.py, seg_type "recording" (its default, no stages):
             same filters, resample to 100 Hz, RobustScaler (median/IQR) with clipping at +-20,
             U-Net, argmax, then merge (both < 0.3 s and gap < 0.1 s) and keep 0.3-2.5 s.
  sumov2_n2  as sumov2 but seg_type "block" (scripts/paper/eval_dreams.py): the signal is cut into
             contiguous N2 blocks of the EXPERT hypnogram (cache/labels/<ds>/<rec>__hypno.csv),
             every block is filtered / resampled / scaled separately. This is the published
             DREAMS protocol; it uses expert staging, so it is not stage-free like the others.
score = mean spindle-class softmax probability over the event's samples.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path(os.environ.get("SPINDLEBENCH_CACHE") or ROOT / "cache")  # as detectors/common.py
RESULTS = Path(os.environ.get("SPINDLEBENCH_RESULTS") or ROOT / "results")
ENVS = {"sumo": HERE / ".venv-sumo", "sumov2": HERE / ".venv-sumov2", "sumov2_n2": HERE / ".venv-sumov2"}
REPOS = {"sumo": HERE / "src" / "sumo", "sumov2": HERE / "src" / "sumov2", "sumov2_n2": HERE / "src" / "sumov2"}
FS = 100  # model sampling rate


def parse_args():
    # same options as common.parse_args plus --model
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(CACHE / "manifest.csv"))
    p.add_argument("--out", default=str(RESULTS / "detections"))
    p.add_argument("--datasets", default="dreams_spindles,nap")
    p.add_argument("--channels", default=None, help="NAP channels, default C3")
    p.add_argument("--recs", default=None)
    p.add_argument("--model", default="sumo,sumov2,sumov2_n2")
    return p.parse_args()


def in_env(env):
    return Path(sys.prefix).resolve() == env.resolve()


def dispatch(args):
    """re-run this script inside the right venv for each model group"""
    groups = {}
    for m in args.model.split(","):
        groups.setdefault(ENVS[m], []).append(m)
    for env, models in groups.items():
        if in_env(env):
            run_models(args, models)
            continue
        cmd = [str(env / "Scripts" / "python.exe"), str(Path(__file__).resolve()),
               "--manifest", args.manifest, "--out", args.out, "--datasets", args.datasets,
               "--model", ",".join(models)]
        if args.channels:
            cmd += ["--channels", args.channels]
        if args.recs:
            cmd += ["--recs", args.recs]
        subprocess.run(cmd, check=True, cwd=os.getcwd())


# ---------------------------------------------------------------- inside a model environment

def load_model(repo):
    import torch
    sys.path.insert(0, str(repo))
    from sumo.config import Config
    from sumo.model import SUMO
    cwd = os.getcwd()
    os.chdir(repo)  # Config resolves config/*.yaml relative to the repo
    try:
        config = Config("predict", create_dirs=False)
    finally:
        os.chdir(cwd)
    # the SUMOv2 checkpoint pickles a pathlib.PosixPath in its hyper-parameters (saved on
    # Linux); map it to WindowsPath while unpickling. Only state_dict is used.
    import pathlib
    posix = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        kw = {} if torch.__version__.startswith("1.") else {"weights_only": False}
        ckpt = torch.load(repo / "output" / "final.ckpt", map_location="cpu", **kw)
    finally:
        pathlib.PosixPath = posix
    model = SUMO(config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def filters(sf, order=10):
    """0.3 Hz high-pass and 30 Hz low-pass, 10th-order Butterworth SOS (applied forward-backward,
    then polyphase resampling to 100 Hz, in the callers). Below 60 Hz the low-pass is moved to (sf-1)/2 as in
    SUMOv2's preprocess(); SUMO v1 has no branch for that (its downsample() refuses sf < 100,
    and a 30 Hz low-pass is impossible at 50 Hz), so DREAMS excerpt 3 (50 Hz) gets the SUMOv2
    rule and is upsampled 50 -> 100 Hz for both models."""
    from scipy.signal import butter
    lp = 30.0 if sf > 60 else (sf - 1) / 2
    return butter(order, 0.3, btype="high", fs=sf, output="sos"), butter(order, lp, btype="low", fs=sf, output="sos")


def preprocess_sumo(x, sf):
    # predict_plain_data.py: butter_bandpass_filter(x, 0.3, 30, sr, 10) (padlen 3*order), downsample, zscore
    import numpy as np
    from scipy.signal import resample_poly, sosfiltfilt
    from scipy.stats import zscore
    hp, lp = filters(sf)
    y = sosfiltfilt(lp, sosfiltfilt(hp, x, padlen=3 * 10), padlen=3 * 10)
    sfi = int(sf)
    g = np.gcd(sfi, FS)
    y = resample_poly(y, FS // g, sfi // g)
    return zscore(y)


def preprocess_sumov2(x, sf):
    # predict_edf_file.py / eval_dreams.py preprocess(): sosfiltfilt with default padlen,
    # resample_poly(d, 100, sf), RobustScaler + clip +-20
    import numpy as np
    from scipy.signal import resample_poly, sosfiltfilt
    from sklearn.preprocessing import RobustScaler
    hp, lp = filters(sf)
    y = sosfiltfilt(lp, sosfiltfilt(hp, x))
    y = resample_poly(y, FS, int(sf))
    y = RobustScaler().fit_transform(y[:, None])
    return np.clip(y, -20, 20).flatten()


def predict(model, y):
    """argmax segments (start, stop in samples @100 Hz) + spindle probability per sample"""
    import numpy as np
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        prob = F.softmax(model(torch.from_numpy(np.ascontiguousarray(y)).float()[None]), dim=1)[0]
    vect = prob.argmax(dim=0).long().numpy()
    p1 = prob[1].numpy()
    diff = np.diff(np.r_[0, vect, 0])  # sumo.data.spindle_vect_to_indices
    return np.c_[np.argwhere(diff == 1), np.argwhere(diff == -1)], p1


def merge_and_filter_spindles(spindles, merge_dur=0.3, merge_dist=0.1, min_dur=0.3, max_dur=2.5):
    """verbatim from src/sumov2/scripts/predict_edf_file.py (columns: start, stop[, extra])"""
    import numpy as np
    sort_idx = np.argsort(spindles[:, 0])
    spindles = spindles[sort_idx]
    durations = spindles[:, 1] - spindles[:, 0]
    distances = spindles[1:, 0] - spindles[:-1, 1]
    to_merge = (durations[:-1] < merge_dur) & (durations[1:] < merge_dur) & (distances < merge_dist)
    spindles[np.r_[to_merge, False], 1] = spindles[np.r_[False, to_merge], 1]
    spindles = spindles[~np.r_[False, to_merge]]
    durations = spindles[:, 1] - spindles[:, 0]
    to_filter = (durations < min_dur) | (durations > max_dur)
    return spindles[~to_filter]


def n2_blocks(dataset, rec, n_sec):
    """contiguous N2 runs of the expert hypnogram -> [(t0, t1)] in seconds"""
    import numpy as np
    import pandas as pd
    h = pd.read_csv(CACHE / "labels" / dataset / f"{rec}__hypno.csv")
    epoch = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    mask = (h.stage.to_numpy() == "N2").astype(int)
    b = np.diff(np.r_[0, mask, 0])
    return [(s * epoch, min(e * epoch, n_sec)) for s, e in zip(np.argwhere(b == 1)[:, 0], np.argwhere(b == -1)[:, 0])
            if s * epoch < n_sec]


def detect(model, name, x, sf, row):
    """-> array (n, 3): onset s, offset s, score"""
    import numpy as np
    if name == "sumo":
        segs = [(0.0, preprocess_sumo(x, sf))]
    elif name == "sumov2":
        segs = [(0.0, preprocess_sumov2(x, sf))]
    else:  # sumov2_n2
        segs = [(t0, preprocess_sumov2(x[int(t0 * sf):int(t1 * sf)], sf))
                for t0, t1 in n2_blocks(row.dataset, row.rec, len(x) / sf)]
    out = []
    for t0, y in segs:
        if len(y) < 64:  # too short for the U-Net's reflection padding
            continue
        idx, p1 = predict(model, y)
        for a, b in idx:
            out.append((t0 + a / FS, t0 + b / FS, float(p1[a:b].mean())))
    ev = np.array(out, float).reshape(-1, 3)
    if name != "sumo" and len(ev):  # the repo function fails on an empty array (no N2 / no events)
        ev = merge_and_filter_spindles(ev)  # merged events keep the first part's score
    return ev


def run_models(args, models):
    import torch
    sys.path.insert(0, str(HERE.parent))
    import common
    torch.set_num_threads(min(14, os.cpu_count() or 1))
    model = load_model(REPOS[models[0]])
    timing = []
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        for name in models:
            rows, t = [], time.perf_counter()
            for ch in chans:
                x = data[ch_names.index(ch)].astype(float)
                for on, off, sc in detect(model, name, x, sf, row):
                    rows.append(dict(onset=round(on, 3), duration=round(off - on, 3), channel=ch,
                                     event="spindle", score=round(sc, 4)))
            dt = time.perf_counter() - t
            n = common.write(args, row, name, rows)
            timing.append(dict(detector=name, dataset=row.dataset, rec=row.rec, n_channels=len(chans),
                               seconds=round(dt, 2), n_events=n))
            print(name, row.dataset, row.rec, f"{len(chans)} ch", n, f"{dt:.1f}s", flush=True)
    import pandas as pd
    f = Path(args.out).parent / "provenance" / "detections" / "sumo_runtime.csv"  # next to the run records
    f.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(timing)
    if f.exists():
        old = pd.read_csv(f)
        key = ["detector", "dataset", "rec"]
        old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([old, df])
    df.to_csv(f, index=False)


if __name__ == "__main__":
    dispatch(parse_args())
