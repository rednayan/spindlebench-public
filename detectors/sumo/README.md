# SUMO and SUMOv2 (learned spindle detectors, authors' weights)

| | SUMO | SUMOv2 |
|---|---|---|
| Paper | Kaulen et al. 2022, Sci Rep 12:7686 | Grieger et al. 2026, Sci Rep 16:16014 |
| Repo | https://github.com/dslaborg/sumo | https://github.com/dslaborg/sumov2 |
| Commit used | `1f1a61c1f576245cb82be39e7548fba4fda79db5` (2024-11-21) | `032a185b8d87225c49ac8dbb1d1501ad5efb3c2d` (2026-06-17) |
| Weights | `src/sumo/output/final.ckpt` (317,987 B), in the repo | `src/sumov2/output/final.ckpt` (322,196 B), in the repo |
| Licence | MIT (code and weights) | MIT (code and weights) |
| Training data | MODA (MASS SS1/SS3, C3), N2 segments; no DREAMS, no NAP | MODA, same split; the paper says DREAMS was not used for training or validation |
| Environment | `.venv-sumo`: Python 3.9.25, torch 1.10.0+cpu, pytorch-lightning 1.5.4, numpy 1.19.5, scipy 1.6.3 (the repo's pins) | `.venv-sumov2`: Python 3.10.20, torch 2.5.1+cpu, pytorch-lightning 2.6.6, numpy 2.2.6, scipy 1.15.3 (the repo leaves versions unpinned) |

No retraining was done. Both models use the authors' released checkpoints with their own inference code paths.

## Setup and run

```
cd detectors/sumo
git clone https://github.com/dslaborg/sumo src/sumo     && git -C src/sumo   checkout 1f1a61c
git clone https://github.com/dslaborg/sumov2 src/sumov2 && git -C src/sumov2 checkout 032a185
uv venv .venv-sumo --python 3.9   && uv pip install --python .venv-sumo/Scripts/python.exe   --index-strategy unsafe-best-match -r requirements-sumo.txt
uv venv .venv-sumov2 --python 3.10 && uv pip install --python .venv-sumov2/Scripts/python.exe --index-strategy unsafe-best-match -r requirements-sumov2.txt
cd ../..
python detectors/sumo/run.py --datasets dreams_spindles
python detectors/sumo/run.py --datasets nap --channels C3,F3,F4,C4,O1,O2,Cz
.venv/Scripts/python.exe detectors/sumo/check_reproduction.py
```

The exact resolved versions are in `requirements-*.lock.txt`. `run.py` can be started with any interpreter: it re-runs itself inside the right venv for each model (`--model sumo,sumov2,sumov2_n2`, all three by default). Runtimes are written to `results/provenance/detections/sumo_runtime.csv` (next to `--out`).

## Detector ids and what each one does

All three take one channel at a time, in µV from the cache. Units do not matter for either model: SUMO z-scores and SUMOv2 robust-scales its input.

| id | Source code path | Preprocessing | Post-processing | Stages used? |
|---|---|---|---|---|
| `sumo` | `scripts/predict_plain_data.py` | Butterworth high-pass 0.3 Hz, then low-pass 30 Hz, both 10th order SOS, forward-backward (`padlen=30`). Polyphase resampling to 100 Hz. **z-score over the whole recording** | U-Net, then a 42-sample (0.42 s) moving average of the logits, then softmax argmax. Each run of ones is one spindle. No duration limits | No |
| `sumov2` | `scripts/predict_edf_file.py`, `-st recording` (the default) | Same filters (default `padlen`), `resample_poly(x, 100, sf)`, then **RobustScaler (median/IQR) clipped at ±20**, over the whole recording | U-Net, moving average, argmax. `merge_and_filter_spindles`: merge two neighbours when both are shorter than 0.3 s and the gap is under 0.1 s, then keep 0.3–2.5 s | No |
| `sumov2_n2` | `scripts/paper/eval_dreams.py`, `-st block` | The signal is cut into contiguous runs of **expert-scored N2** from `cache/labels/<ds>/<rec>__hypno.csv` (DREAMS: 5 s epochs, NAP: 30 s epochs). Each block is filtered, resampled and robust-scaled on its own | As `sumov2`, over all blocks pooled | **Yes, expert hypnogram** |

`score` is the mean spindle-class softmax probability over the event's samples. A merged SUMOv2 event keeps the score of its first part.

`sumo` and `sumov2` are stage-free, like YASA and A7. Use them in the benchmark tables. `sumov2_n2` is the published DREAMS protocol. It relies on expert staging, so it gets an advantage the other detectors don't have. It is kept for the reproduction check. Five NAP naps have no expert N2 (suj12_d1, suj19_d2, suj20_d2, suj26_d2, suj27_d1), so `sumov2_n2` writes header-only files for them.

SUMOv2 needs no other channels and no hypnogram unless you use block mode. The 20th-order filter mentioned in the SUMOv2 paper belongs to the MODA training preparation (`prepare_moda_data.py`). Both inference scripts use 10th order, and so does this adapter.

## Deviations from the authors' code

1. **DREAMS excerpt 3 is sampled at 50 Hz.** At that rate a 30 Hz low-pass can't be built (Nyquist is 25 Hz), and SUMO v1's `downsample()` refuses to upsample. For both models the adapter uses SUMOv2's own rule: low-pass at (sf−1)/2 = 24.5 Hz when sf ≤ 60, then `resample_poly` upsamples 50 → 100 Hz. For SUMOv2 this is exactly the repo's code. For SUMO it is an extrapolation.
2. **Inference skips the Lightning `Trainer`.** The adapter calls `model.eval()` and the forward pass directly, which is the same computation as `predict_step`: `softmax(self(x)).argmax`. BatchNorm runs in eval mode.
3. **The SUMOv2 checkpoint pickles a `pathlib.PosixPath`**, so it can't be unpickled on Windows as-is. During `torch.load` the adapter maps it to `WindowsPath`. Only `state_dict` is used.
4. **`merge_and_filter_spindles` crashes on an empty array** (a recording with no N2 blocks). Such recordings skip the call.

The warning `Using padding='same' with even kernel lengths and odd dilation` is harmless. It comes from the authors' model definition.

## Reproduction check (SUMOv2, DREAMS)

`check_reproduction.py` copies the scoring code from `src/sumov2/scripts/final/dreams-f1-vs-experts.ipynb`:

- Detections and labels are kept when their midpoint falls in an expert N2 epoch.
- A match needs IoU > 0.2, with greedy many-to-one matching.
- Counts are pooled over excerpts 1–8 for expert 1 and 1–6 for expert 2.
- The whole recording is scored. There is no 1000 s window for expert 1.

The script uses the cache's labels and hypnograms.

| Predictions | vs expert 1 (P / R / F1) | vs expert 2 (P / R / F1) |
|---|---|---|
| Published (paper / notebook) | 0.479 / 0.599 / **0.532** | **0.76** |
| Repo's shipped predictions `output/annotations/dreams/*.npz`, our labels | 0.479 / 0.599 / **0.532** | 0.890 / 0.670 / **0.764** |
| **Ours, `sumov2_n2`** | 0.479 / 0.599 / **0.532** | 0.890 / 0.670 / **0.764** |
| Ours, `sumov2` (stage-free) | 0.478 / 0.595 / 0.531 | 0.894 / 0.667 / 0.764 |
| Ours, `sumo` (stage-free) | 0.507 / 0.488 / 0.497 | 0.942 / 0.514 / 0.665 |

The reproduction is exact. For all 8 excerpts, `sumov2_n2` produces the same number of events as the repo's shipped predictions (60, 48, 30, 11, 44, 44, 24, 113). Every event is matched at IoU > 0.8, and onsets differ by at most 0.01 s, which is rounding.

With the thesis's `experiments/rq1_benchmark/evaluate.py` (`rule=iou tol=0.2 scope=N2`, one-to-one Hungarian matching), the numbers are as follows:

| Reference | `sumov2` | `sumov2_n2` | `sumo` |
|---|---|---|---|
| expert 1 | 0.709 | 0.712 | 0.633 |
| expert 2 | 0.764 | 0.764 | 0.665 |

The expert-1 figure is higher here because `experiments/rq1_benchmark/evaluate.py` scores expert 1 only inside its 0–1000 s window. The published 0.53 counts every detection after 1000 s as a false positive, even though V1 did not score that part.

## Main-protocol results (`experiments/rq1_benchmark/evaluate.py --detectors sumo,sumov2,yasa_default,a7 --datasets dreams_spindles,nap`)

**DREAMS**, onset ±0.5 s, vs expert 1, pooled:

| Scope | `sumov2` | `a7` | `sumo` | `yasa_default` |
|---|---|---|---|---|
| all | **0.649** | 0.598 | 0.561 | 0.402 |
| N2 | 0.693 | 0.668 | 0.620 | 0.422 |

**NAP**, centre ±0.5 s, vs expert, C3, 30 naps not excluded:

| Scope | `a7` | `sumov2` | `sumo` | `yasa_default` |
|---|---|---|---|---|
| all | 0.274 | 0.271 | 0.257 | 0.219 |
| N2 | 0.304 | 0.309 | 0.284 | 0.231 |

On NAP every detector scores low, and the learned models land on the same level as A7. The NAP labels are point marks. On C3 the median difference between a detection centre and the nearest expert mark (within 2 s) is +0.06 s for `sumov2`, +0.09 s for `sumo` and +0.08 s for A7. So there is no timing offset in the adapter. The gap comes from the reference, not from units or sampling rate: F1 rises to about 0.37 at ±1 s and about 0.40 at ±1.5 s for all of them.

## Runtime (CPU, 14 cores; model time only, excluding loading the `.npz`)

| id | DREAMS, 1 channel, 30 min | NAP, 7 channels, ~30 min |
|---|---|---|
| `sumo` | 0.16 s per recording | 1.1 s per recording |
| `sumov2` | 0.11 s | 0.8 s |
| `sumov2_n2` | 0.14 s | 0.35 s |

A full run of both datasets takes about 2 minutes of wall time, mostly spent loading the 63-channel NAP cache files.

## Caveats

- **Domain.** Both models were trained only on MODA: C3, N2, healthy adults, 115 s segments, with labels from the MODA consensus. DREAMS and NAP were never seen in training, so DREAMS rows are zero-shot. On NAP the models also run on F3, F4, C4, O1, O2 and Cz, which they never saw in training.
- **Whole-recording normalisation.** `sumo` and `sumov2` normalise over the whole recording, including wake and artefacts. SUMO's z-score is sensitive to large artefacts. SUMOv2's median/IQR scaling with clipping is robust to them. SUMO was trained on N2-only segments, so normalising over a whole nap is the authors' plain-data use, not their training condition.
- **No duration limits for SUMO.** SUMO v1 has no minimum or maximum duration, so its detections are raw argmax runs (up to 15.4 s on NAP).
- **Excerpt 3 (50 Hz)** is upsampled to 100 Hz; see the deviations above. The paper shows SUMOv2 does poorly on it (F1 0.23 vs expert 1). Our output matches.
- **Versions.** The SUMOv2 environment uses current library versions, because the repo pins none. The event-level match with the shipped predictions shows that this does not change the output.
