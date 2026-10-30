# S5 (detector ids `s5`, `s5_ens`)

*S5: Self-Supervised Learning Boosts Sleep Spindle Detection in Single-Channel EEG via Temporal
Segmentation.* IEEE Trans Biomed Eng (2026), doi:10.1109/TBME.2026.3675979. The full text was not
accessible on 2026-10-30, so the paper's own numbers are not checked here.

```
git clone https://github.com/Adrenaline613/S5 detectors/s5/src     # commit e31fb2e, 18 Mar 2026
git -C detectors/s5/src checkout e31fb2ef00bca688249c11ce06dedd3c570ed12c
cd detectors/s5 && uv sync                                          # Python 3.9, torch 2.3.1+cpu
cd ../.. && detectors/s5/.venv/Scripts/python detectors/s5/run.py \
    --datasets dreams_spindles,nap --channels C3,F3,F4,C4,O1,O2,Cz
```

## Source and weights
| what | where |
|---|---|
| code | [Adrenaline613/S5](https://github.com/Adrenaline613/S5), GPL-3.0, cloned unmodified into `src/` (ignored by git) |
| weights | `src/weight/MODA_S3Net_fullyfinetune_20250715/fold1..5_best.ckpt`, in the repository |
| training data | MODA consensus spindles on MASS segments, five folds. DREAMS and NAP are not mentioned anywhere in the repository, so both scores are treated as out-of-sample |

## Model
S3Net: a dilated residual 1-D CNN encoder with an ASPP module and a decoder, giving a per-sample
2-class output at 100 Hz, smoothed by a 0.42 s moving average. Self-supervised pretraining, then
full fine-tuning on MODA ("fullyfinetune" in the checkpoint name). Fully convolutional, so the input
length is free.

## Pipeline (`run.py`), following the authors' `inference.py`
1. One channel in µV, **no normalisation**. The MODA training export also stores raw µV
   (`moda_to_numpy.py` is called with `is_norm=False`).
2. Butterworth high-pass 1 Hz and low-pass 30 Hz, order 10, `sosfiltfilt`, as in `inference.py`
   (the training export used 0.3-30 Hz; immaterial for spindles). The low-pass is capped at
   0.45 x the sampling rate, which only affects DREAMS excerpt 3 (50 Hz).
3. Resample to 100 Hz with `resample_poly`.
4. Consecutive 30 s epochs from t = 0, as in `inference.py`. The authors drop the tail; here one
   end-aligned extra epoch covers it and only its uncovered samples are kept.
5. Argmax per sample, then the authors' `process_events` per epoch with the `inference.py` settings
   (min 0.5 s, max 3 s, merge gaps ≤ 0.3 s). Events at epoch borders can be split, as in the
   authors' code. Stage-free: every epoch is run (`inference.py` keeps N2 epochs only; the evaluator
   applies the N2 scope instead).

| id | weights | note |
|---|---|---|
| `s5` | fold 1 | as in `inference.py` |
| `s5_ens` | mean softmax of folds 1-5 | our ensemble of the released folds |

`score` = mean spindle probability over the event.

## Runtime (14-core CPU)
DREAMS (8 excerpts, 1 channel) plus NAP (36 naps x 7 channels), both ids: about 5 min.

## Sanity check (pooled F1, N2, `experiments/rq1_benchmark/evaluate.py`)
| reference | protocol | `s5` P / R / F1 | `s5_ens` F1 |
|---|---|---|---|
| DREAMS expert 1 | onset ±0.5 s | 0.77 / 0.62 / **0.68** | 0.66 |
| DREAMS expert 2 | onset ±0.5 s | 0.80 / 0.67 / **0.73** | 0.72 |
| NAP expert | centre ±0.5 s | 0.40 / 0.24 / **0.30** | 0.30 |

Second among out-of-sample detectors on DREAMS expert 1, behind SUMOv2 (0.69).
