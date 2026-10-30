# SSSM, Sleep Semantic Segmentation Model (detector ids `sssm`, `sssm_strict`, `sssm_v2`)

Bao X, Li M, Chen D, Wang Z, Huang Q, Wen Z, Li Y. *Semantic Segmentation of Sleep Events for
High-Resolution Sleep Decoding.* medRxiv 2025, doi:10.1101/2025.09.25.25336636 (preprint, not peer reviewed).

```
cd detectors/sssm && uv sync                     # Python 3.12, torch 2.14.1+cpu from the PyTorch CPU index
cd ../.. && detectors/sssm/.venv/Scripts/python detectors/sssm/run.py \
    --datasets dreams_spindles,dreams_kc,nap --channels C3,F3,F4,C4,O1,O2,Cz
```

## Source and weights
| what | where |
|---|---|
| model package (used here) | PyPI `sssm==0.1.0` (wheel uploaded 30 Jun 2026; README cites the preprint). It contains `sssm.Model`, the CNN and three checkpoints. |
| GitHub | https://github.com/didi226/Sleep-Semantic-Segmentation, commit `305efb04e2d8f5e986f7f52824c06718cac53ed6` (28 Sep 2024, `sssm` 0.0.6 plus the `wrap_sssm` event-detection wrapper, author Xiaoyu Bao). It has not been updated for 0.1.0. Docs: https://sleep-semantic-segmentation.readthedocs.io |
| weights | `sssm/saved_models/model.pt` (package default, md5 `37ea62bf…`, file date May 2024, identical to the 0.0.6 GitHub checkpoint), `model_V1.pt` (May 2024, unused), `model_V2.pt` (md5 `e827738c…`, file date **25 Sep 2025, the day the preprint was submitted**) |
| licence | **None declared.** There is no LICENSE file in the repository and no licence field on PyPI. Legally that means all rights reserved. It can be used for research benchmarking, but the weights shouldn't be redistributed. Cite the preprint. |

The paper's full text sits behind medRxiv's bot protection and couldn't be fetched. These points come from the abstract, the package README and secondary summaries:
- the training data is about 276k events from MASS, Sleep-EDF, ISRUC and CAP (built with self-supervised and active learning);
- **DREAMS is reported as an independent, unseen test set** (paper F1 about 0.90 for spindles and K-complexes under the paper's own matching rule);
- MODA (MASS) was used for spindle validation.

Which checkpoint the paper's figures used is **not documented**.

## Model
- The model is a 3-conv-block 1D CNN (TS-TCC style encoder) that classifies a **3 s window (300 samples at 100 Hz)** into 7 classes: Spindle, K-complex, Slow wave, Sawtooth, Vertex Sharp, Background, Arousal.
- "Semantic segmentation" means this window classifier is slid over the signal. The class probability is assigned to the window centre.
- There is **no input normalisation**. The model expects µV and is amplitude-sensitive: scaling DREAMS by 0.5× turns almost everything into Background, and 2× turns it into Slow wave or Arousal.

## Pipeline (`run.py`)
1. One channel in µV. FIR band-pass 0.1–40 Hz (`mne.filter.filter_data` defaults), as in the authors' `wrap_sssm` example. The low-pass is dropped when 40 Hz ≥ Nyquist, which only affects DREAMS excerpt 3 at 50 Hz. The authors' other notebook uses 0.7–45 Hz; the difference is immaterial for 3 s windows.
2. Resample to **100 Hz** (`mne.filter.resample`) and cast to float16, as the wrapper does.
3. Slide the windows with **step 50 samples (0.5 s)**, the `wrap_sssm.sleep_event_detect` default. Softmax over the 7 classes.
4. Post-processing with the authors' `Model.to_pandas`:
   - consecutive windows whose probability for the same class is > 0.5 are merged into one event;
   - the event spans samples `[first·50 + 125, last·50 + 175]`, so onsets lie on a 0.5 s grid and the minimum duration is 0.5 s;
   - the event is kept if its mean probability ≥ the per-class threshold.
5. Output: `Spindle → event=spindle`, `K-complex → event=kcomplex`, `score =` mean class probability. The other five classes are dropped.

| id | weights | per-class threshold | source of the setting |
|---|---|---|---|
| `sssm` | `model.pt` (package default) | 0.5 | `sssm` 0.1.0 `Model.to_pandas` default |
| `sssm_strict` | `model.pt` | Spindle 0.95, K-complex 0.6 | `wrap_sssm.sleep_event_detect` defaults (GitHub, 2024) |
| `sssm_v2` | `model_V2.pt` | 0.5 | the checkpoint dated with the preprint; same post-processing as `sssm` |

Nothing was tuned. All three settings come from the authors' code. `sssm_v2` is included because it is the likely paper checkpoint, even though the package still defaults to `model.pt`.

## Runtime (14-core CPU, no GPU)
Not recorded when the adapter was run. TODO: time one full run of `run.py` and fill this in.

## Sanity check (main venv, `experiments/rq1_benchmark/evaluate.py`, pooled F1)
DREAMS K-complexes, all 10 excerpts, against expert 1, onset ±0.5 s, N2 by event midpoint (independent
rescore during the audit, 2026-10-27; it matched `results/benchmark/summary.csv` at that time):

| id | F1 |
|---|---|
| `sssm` | 0.406 |
| `sssm_strict` | 0.356 |
| `sssm_v2` | 0.250 |

Spindle rows (DREAMS, NAP): see `results/benchmark/summary.csv`. TODO: copy them here once the current
recompute of `experiments/rq1_benchmark/evaluate.py` has finished.

## Caveats
- **No licence.** Use for research benchmarking only; do not redistribute the weights.
- **Unclear checkpoint.** Which of `model.pt` / `model_V2.pt` the paper's figures used is not
  documented; `sssm_v2` is only the likely one.
- **Amplitude sensitivity.** There is no input normalisation (see Model), so results depend on
  the µV scale of each dataset's preprocessing.
- **No duration cap.** The post-processing has no maximum duration by design. 13–14% of spindle
  events are longer than 3 s, up to 31 s on NAP.
- **Training data unverified.** The source datasets come from secondary summaries, so overlap with
  DREAMS or MASS cannot be ruled out.
