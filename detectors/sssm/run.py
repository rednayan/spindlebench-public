"""
SSSM, Sleep Semantic Segmentation Model (Bao et al. 2025, medRxiv 10.1101/2025.09.25.25336636),
authors' weights from the `sssm` 0.1.0 PyPI package.

    detectors/sssm/.venv/Scripts/python detectors/sssm/run.py \
        [--datasets dreams_spindles,dreams_kc,nap] [--channels C3,F3,...] [--recs ...]
        [--model model.pt --id sssm]

Pipeline (authors' `sssm` 0.1.0 model class; post-processing = its Model.to_pandas):
  1. single channel in µV; FIR band-pass 0.1-40 Hz (mne defaults, as in the authors' wrap_sssm
     example; low-pass dropped when 40 Hz >= Nyquist, i.e. DREAMS excerpt 3 at 50 Hz)
  2. resample to 100 Hz (mne FFT resample); cast to float16 as the wrapper does. No z-scoring:
     the model is amplitude-sensitive and expects µV
  3. 3 s windows (300 samples) slid with step 50 samples (0.5 s, the wrapper default); 7-class
     softmax per window (Spindle, K-complex, Slow wave, Sawtooth, Vertex Sharp, Background, Arousal)
  4. Model.to_pandas: runs of consecutive windows whose same-class probability > 0.5 form one
     event spanning samples [first*step + 125, last*step + 175]; the event is kept if its mean
     probability >= the per-class threshold
Configurations written (all documented by the authors, none tuned here):
  sssm         model.pt (package default weights), per-class threshold 0.5 (sssm 0.1.0 to_pandas default)
  sssm_strict  model.pt, wrap_sssm sleep_event_detect defaults (Spindle 0.95, K-complex 0.6)
  sssm_v2      model_V2.pt (also shipped in sssm 0.1.0, dated 2025-09-25), threshold 0.5
Emits event = spindle (Spindle) and event = kcomplex (K-complex); score = mean class probability.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import mne  # noqa: E402
import torch  # noqa: E402
import sssm  # noqa: E402

FS = 100
STEP = 50
OVERALL = 0.5
CLASSES = ["Spindle", "K-complex", "Slow wave", "Sawtooth", "Vertex Sharp", "Background", "Arousal"]
TH_PACKAGE = {c: 0.5 for c in CLASSES}  # sssm 0.1.0 Model.to_pandas default
TH_WRAPPER = {"Spindle": 0.95, "Background": 0.9, "Arousal": 0.9, "K-complex": 0.6,  # wrap_sssm defaults
              "Slow wave": 0.6, "Vertex Sharp": 0.6, "Sawtooth": 0.6}
CONFIGS = {"sssm": ("model.pt", TH_PACKAGE), "sssm_strict": ("model.pt", TH_WRAPPER),
           "sssm_v2": ("model_V2.pt", TH_PACKAGE)}
EMIT = {"Spindle": "spindle", "K-complex": "kcomplex"}


def preprocess(x, sf):
    h = 40.0 if 40.0 < sf / 2 else None
    x = mne.filter.filter_data(x[None].astype(np.float64), sf, 0.1, h, method="fir", verbose=False)[0]
    if sf != FS:
        x = mne.filter.resample(x, up=FS / sf, down=1.0, npad="auto", verbose=False)
    return x


def events(model, th):
    try:
        df = model.to_pandas(overall_threshold=OVERALL, event_threshold=dict(th))
    except IndexError:  # no window above the overall threshold
        return []
    df = df[df.label.isin(EMIT)]
    return [(r.Start / FS, (r.End - r.Start) / FS, EMIT[r.label], float(r.predict_proba)) for r in df.itertuples()]


if __name__ == "__main__":
    args = common.parse_args(["dreams_spindles", "dreams_kc", "nap"])
    torch.set_num_threads(8)
    models = {w: sssm.Model(device="cpu", model_name=w) for w in {w for w, _ in CONFIGS.values()}}
    t_all = time.time()
    for row, data, sf, ch_names, chans in common.iter_recordings(args):
        t0 = time.time()
        out = {k: [] for k in CONFIGS}
        for ch in chans:
            y = preprocess(data[ch_names.index(ch)], sf).astype(np.float16)[None, :]
            for w, model in models.items():
                model.predict(y, step=STEP)
                for name, (wn, th) in CONFIGS.items():
                    if wn != w:
                        continue
                    for o, d, ev, c in events(model, th):
                        out[name].append(dict(onset=round(o, 3), duration=round(d, 3), channel=ch, event=ev,
                                              score=round(c, 4)))
        msg = []
        for name, rows in out.items():
            common.write(args, row, name, rows)
            ns = sum(r["event"] == "spindle" for r in rows)
            msg.append(f"{name} {ns} sp/{len(rows) - ns} kc")
        print(f"{row.dataset} {row.rec} {len(chans)} ch | {' | '.join(msg)} | {time.time() - t0:.1f}s", flush=True)
    print(f"total {time.time() - t_all:.0f}s")
