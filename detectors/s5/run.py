"""
S5 (S3Net), self-supervised spindle segmenter, authors' repository and MODA weights (GPL-3.0).

    detectors/s5/.venv/Scripts/python detectors/s5/run.py [--datasets dreams_spindles,nap] [--channels C3,...]

Pipeline as in the authors' inference.py (src/inference.py, commit e31fb2e):
  1. one channel in uV, no normalisation (the MODA training export also stores raw uV, is_norm=False);
  2. Butterworth high-pass 1 Hz and low-pass 30 Hz, order 10, zero-phase (sosfiltfilt). The low-pass
     is capped at 0.45 x the sampling rate (only DREAMS excerpt 3, 50 Hz);
  3. resample to 100 Hz (resample_poly);
  4. consecutive 30 s epochs from t = 0. The authors drop the leftover tail; here one extra 30 s
     epoch aligned to the end covers it, and only its samples after the last full epoch are kept;
  5. per-sample argmax over 2 classes, then process_events per epoch (min 0.5 s, max 3 s, merge
     gaps <= 0.3 s), the settings of inference.py. Stage-free: every epoch, not only N2.
Detector ids:
  s5      fold-1 checkpoint, as in the authors' inference.py
  s5_ens  mean softmax of the five fold checkpoints, then the same argmax and post-processing
score = mean spindle probability over the event.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
from model.net import S3Net  # noqa: E402
from model.postprocessing import process_events  # noqa: E402

FS = 100
EPOCH = 30 * FS
WEIGHTS = sorted((HERE / "src/weight/MODA_S3Net_fullyfinetune_20250715").glob("fold*_best.ckpt"))
torch.set_num_threads(8)


def load(ckpt):
    m = S3Net(in_channels=1, num_classes=2, training_mode="scratch")
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    return m.eval()


MODELS = [load(w) for w in WEIGHTS]


def preprocess(x, sf):
    hp = butter(10, 1, btype="hp", fs=sf, output="sos")
    lp_ = butter(10, min(30, 0.45 * sf), btype="lp", fs=sf, output="sos")
    x = sosfiltfilt(lp_, sosfiltfilt(hp, x, padlen=30), padlen=30)
    sf, g = int(sf), np.gcd(int(sf), FS)
    return x if sf == FS else resample_poly(x, FS // g, sf // g)


def epochs(x):
    n = len(x) // EPOCH
    starts = [i * EPOCH for i in range(n)]
    if len(x) > n * EPOCH and len(x) >= EPOCH:
        starts.append(len(x) - EPOCH)  # tail epoch, end-aligned
    return starts


@torch.no_grad()
def prob(models, x, starts):
    batch = torch.from_numpy(np.stack([x[s:s + EPOCH] for s in starts]).astype(np.float32))[:, None]
    return np.mean([torch.softmax(m(batch), 1)[:, 1].numpy() for m in models], axis=0)  # (n_epochs, EPOCH)


def events(p, starts, n_samples, ch):
    lab = np.zeros(n_samples, np.int8)
    pr = np.zeros(n_samples)
    covered = 0
    for s, pe in zip(starts, p):
        e = process_events((pe > 0.5).astype(np.int64), FS, 0.5, 3, 0.3)
        lo = max(covered - s, 0)  # tail epoch: keep only the uncovered part
        lab[s + lo:s + EPOCH] = e[lo:]
        pr[s + lo:s + EPOCH] = pe[lo:]
        covered = s + EPOCH
    d = np.diff(np.r_[0, lab, 0])
    on, off = np.where(d == 1)[0], np.where(d == -1)[0]
    return [dict(onset=a / FS, duration=(b - a) / FS, channel=ch, event="spindle",
                 score=round(float(pr[a:b].mean()), 4)) for a, b in zip(on, off)]


if __name__ == "__main__":
    args = common.parse_args(["dreams_spindles", "nap"])
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        out = {"s5": [], "s5_ens": []}
        for ch in chans:
            x = preprocess(data[ch_names.index(ch)].astype(float), sf)
            starts = epochs(x)
            if not starts:
                continue
            out["s5"] += events(prob(MODELS[:1], x, starts), starts, len(x), ch)
            out["s5_ens"] += events(prob(MODELS, x, starts), starts, len(x), ch)
        n = {k: common.write(args, row, k, v) for k, v in out.items()}
        print(row.dataset, row.rec, n, flush=True)
