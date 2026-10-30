# A7 spindle detector (Lacourse et al. 2019)

Lacourse, Delfrate, Beaudry, Peppard & Warby (2019). *A sleep spindle detection algorithm that
emulates human expert spindle scoring.* J Neurosci Methods 316:3–11. doi:10.1016/j.jneumeth.2018.08.014

```
.venv/Scripts/python.exe detectors/a7/run.py --datasets dreams_spindles,nap [--channels C3,F3,F4,C4,O1,O2,Cz]
```
It runs in the main `.venv` (needs only numpy, scipy and pandas).

| id | what |
|---|---|
| `a7` | every detection the official code writes to `EventDetection.txt` (published defaults) |
| `a7_incontext` | only the detections the spectral-context classifier marks IN context (`contextClassifier = 1`) |

## Source
- **Official code (MATLAB):** https://github.com/swarby/A7_LacourseSpindleDetector (Warby lab), commit
  `e67b6372c96b4d242ba96c18eb6b68d3266bf605` (2021-04-23), release 1.1.2. Licence: GPL-3.0 ("free use and
  modification permitted provided modifications are also freely distributed"; cite the paper).
- **No official stand-alone Python version exists.** The CEAMS Snooz toolbox
  (https://github.com/SnoozToolbox/snooz-package-ceams, `modules/CEAMSModules/SpindleDetectorA7`, GPL-3.0,
  commit `270505ec`) has a Python A7 module. It is a reimplementation inside a GUI framework, and it differs
  from the MATLAB code: the baseline is not restricted to a sleep stage, and there's no context classifier.
  So **`a7.py` is a line-by-line port of the official MATLAB code**. Snooz was used only to cross-check the
  preprocessing (resample to 100 Hz, 0.3–30 Hz band-pass) and the default thresholds.

## Parameters (published defaults, `a7MainScript.m` / `initA7_DEF.m`)
- Sampling rate 100 Hz. Windows of 0.3 s with a 0.1 s step. Zero-padded 2-s FFT, Hann window, `IntegSpectPow` normalisation.
- absSigPow: log10 of the mean squared 11–16 Hz signal (Butterworth order 20, HP then LP, zero phase) > **1.25** (log10 µV²).
- relSigPow: log10(PSD 11–16 / PSD 4.5–30), z-scored against a 30-s baseline, > **1.6**.
- sigmaCov: log10(max(cov(raw, sigma), 0) + 1), z-scored against a 30-s baseline, > **1.3**.
- sigmaCorr: Pearson correlation of raw and sigma > **0.69**.
- Baseline: the 30 s of valid windows nearest each window. Only values between the 10th and 90th
  percentile count. Valid = sleep stage **N2** and no artifact.
- An event starts where all 4 features exceed their thresholds. It is then extended while absSigPow and
  sigmaCov stay above theirs. Duration must be 0.3–2.5 s.
- Context classifier: log10(mean PSD 0.5–8 / mean PSD 16–30 over the 30-s baseline + 1) > 0.9.

## Adapter decisions
- **Preprocessing:** polyphase resample to 100 Hz, then a zero-phase Butterworth 0.3–30 Hz band-pass
  (order 5 sos, as in the Snooz A7 tool). The official example input is "C3 filtered 0–30 Hz" at 100 Hz.
- **DREAMS excerpt 3 (50 Hz) is upsampled to 100 Hz.** All filters then stay below Nyquist. The signal has
  no content above 25 Hz, so the 4.5–30 Hz total power used by relSigPow and the 16–30 Hz band of the context
  classifier are under-filled. The z-scoring against the same recording's baseline absorbs most of this.
- **Baseline stage.** The ranked runs `a7` and `a7_incontext` use the whole recording as the baseline (the
  code's "no staging" option), like every other benchmark detector, which runs without a hypnogram. The
  published configuration (`bslSleepStaging = 'N2'`, baseline = N2 epochs of the expert hypnogram
  `cache/labels/<ds>/<rec>__hypno.csv`) is kept as `a7_n2base` / `a7_incontext_n2base`. They use expert
  staging, so they are never ranked (`spindle_eval.USES_EXPERT_STAGING`). The difference is small:
  N2 F1 0.670 vs 0.668 on DREAMS and 0.318 vs 0.300 on NAP (stage-free vs N2 baseline). Detection runs on
  the whole recording either way; the evaluator applies its own N2 scope. No artifact vector is
  available, so every sample counts as artifact-free.
- **Onsets** are `(start_sample - 1)/100` in MATLAB terms, i.e. the 0-based sample / 100. Durations are
  `(end - start + 1)/100`. `score` is empty.

## Port fidelity
- **Checked against the official example data** (`input/EEGvector.mat`, 60 s of N2). All 9 events of
  `output/EventDetection_example.txt` are reproduced exactly: same start and end samples, all IN context.
  The 4 feature time series in `detectionInfo_example.mat` match to < 1e-12 away from the recording edges.
  Within 5 s of the edges the differences are up to 0.03. They come from the filter edge padding: MATLAB
  `filtfilt` uses a different pad length than scipy's `sosfiltfilt`.
- **Known quirks kept:** MATLAB `prctile` interpolation (numpy `method="hazen"`); `std` with ddof=1; the
  `iAvailable(end)` index/position mix-up in the baseline loop; and the re-extraction of events from the
  detection vector, which merges touching events.
- **Not replicated:**
  - In `a7subSigmaCov.m`, when fewer than 30 s of baseline exist, `nWindows` is overwritten. That is an edge
    case that never occurs here.
  - The output's context flag is computed per final (merged) event. MATLAB computes it per event before
    merging and can misalign rows when a merge happens.

## Runtime
CPU, single process: about 0.5 s per DREAMS excerpt and 6–9 s per NAP recording for 7 channels. All 36 NAP
recordings × 7 channels take about 4 min.

## Sanity check (experiments/rq1_benchmark/evaluate.py, main protocol)
DREAMS spindles, onset ±0.5 s vs expert 1, pooled over 8 excerpts:

| detector | scope | P | R | F1 |
|---|---|---|---|---|
| a7 | N2 | 0.750 | 0.602 | **0.668** |
| a7_incontext | N2 | 0.748 | 0.565 | 0.644 |
| dreams_auto | N2 | 0.666 | 0.652 | 0.659 |
| wonambi_lacourse2018 (independent reimplementation) | N2 | 0.765 | 0.579 | 0.659 |
| yasa_default | N2 | 0.911 | 0.274 | 0.422 |

NAP, centre ±0.5 s vs the expert, C3, 30 included recordings: a7 N2 F1 0.304 (P 0.399, R 0.245);
a7_incontext 0.294; yasa_default 0.231.

The context classifier removes many detections on excerpts 4 (124 → 46) and 8 (128 → 76), and almost none
elsewhere. It rejects events in stretches with a low (0.5–8 Hz)/(16–30 Hz) power ratio, i.e. wake-like or
high-beta EEG. Excerpt 4 is the one whose staging is suspect (main README).
