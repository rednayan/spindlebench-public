"""
OpenSpindleNet (Sejak et al. 2025, Comput Biol Med 197:110854), scalp-EEG ONNX model, authors' weights.

    detectors/openspindlenet/.venv/Scripts/python detectors/openspindlenet/run.py \
        [--datasets dreams_spindles,nap] [--channels C3,F3,...] [--recs ...]

Pipeline (as published / as in the authors' `openspindlenet` 0.1.1 package):
  1. resample the channel to 250 Hz by linear interpolation (as in the authors' DREAMS
     export, mayo_spindles/dreams_to_hdf5.py); no extra filtering
  2. cut into consecutive, non-overlapping 30 s blocks (7500 samples) from t = 0; the
     remaining tail is covered by one extra block aligned to the end of the recording,
     from which only detections centred in the uncovered tail are kept
  3. per block: z-score raw signal; |CWT| (shan6-13, 15 scales geomspace(135, 270)) z-scored;
     ONNX model -> detection head [30 cells x (conf, centre offset, duration)]
  4. package post-processing: sigmoid, confidence >= 0.5, interval NMS IoU 0.3
Writes detector id `openspindlenet` (event = spindle, score = detection confidence).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import onnxruntime as ort  # noqa: E402
import pywt  # noqa: E402
from openspindlenet import get_model_path  # noqa: E402
from openspindlenet.evaluator import Evaluator  # noqa: E402

DETECTOR = "openspindlenet"
FS = 250
BLOCK = 30 * FS
CONF, NMS_IOU = 0.5, 0.3
SCALES = np.geomspace(135, 270, num=15)  # openspindlenet/inference.py __convert_to_scalogram
BATCH = 64


def znorm(a):
    """openspindlenet/inference.py __normalize: z-score along the last axis, NaN -> 0"""
    return np.nan_to_num((a - a.mean(-1, keepdims=True)) / a.std(-1, keepdims=True), nan=0.0)


def run_blocks(sess, blocks):
    """blocks [n, 7500] -> list of [k, 3] intervals (start, end, conf) in samples, per block.
    Same as SpindleInference.run(), batched; the CWT uses method='fft' (identical to the
    package's direct convolution to ~2e-6 relative, 30x faster). Input stays float32 as in the
    package and in the authors' training export: pywt's float32 CWT differs by ~9% from float64."""
    out = []
    for i in range(0, len(blocks), BATCH):
        b = blocks[i:i + BATCH]
        raw = znorm(b[:, None, :]).astype(np.float32)
        scal = np.abs(pywt.cwt(b, SCALES, "shan6-13", sampling_period=1 / FS, method="fft", axis=-1)[0])
        scal = znorm(np.moveaxis(scal, 0, 1)).astype(np.float32)  # [n, 15, 7500]
        det = sess.run(["detection"], {"raw_signal": raw, "spectrogram": scal})[0]
        det = 1 / (1 + np.exp(-det))
        for d in det:
            iv = Evaluator.detections_to_intervals(d, BLOCK, confidence_threshold=CONF)
            out.append(Evaluator.intervals_nms(iv, iou_threshold=NMS_IOU))
    return out


def resample_linear(x, sf):
    n = int(round(len(x) * FS / sf))
    return np.interp(np.arange(n) / FS, np.arange(len(x)) / sf, x).astype(np.float32)


def detect_channel(sess, x, sf):
    y = resample_linear(x, sf)
    n = len(y)
    starts = list(range(0, n - BLOCK + 1, BLOCK))
    covered = starts[-1] + BLOCK if starts else 0
    tail = covered < n and n >= BLOCK
    if tail:
        starts.append(n - BLOCK)
    out = []
    if not starts:
        return out
    per_block = run_blocks(sess, np.stack([y[s:s + BLOCK] for s in starts]).astype(np.float32))
    for k, (s, ivs) in enumerate(zip(starts, per_block)):
        for a, b, c in ivs:
            if tail and k == len(starts) - 1 and s + (a + b) / 2 < covered:
                continue
            out.append((float(s + a) / FS, float(b - a) / FS, float(c)))
    return out


if __name__ == "__main__":
    args = common.parse_args(["dreams_spindles", "nap"])
    sess = ort.InferenceSession(get_model_path("eeg"), providers=["CPUExecutionProvider"])
    t_all = time.time()
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        t0 = time.time()
        rows = []
        for ch in chans:
            for o, d, c in detect_channel(sess, data[ch_names.index(ch)].astype(np.float64), sf):
                rows.append(dict(onset=round(o, 4), duration=round(d, 4), channel=ch, event="spindle",
                                 score=round(c, 4)))
        n = common.write(args, row, DETECTOR, rows)
        print(f"{row.dataset} {row.rec} {len(chans)} ch {n} spindles {time.time() - t0:.1f}s", flush=True)
    print(f"total {time.time() - t_all:.0f}s")
