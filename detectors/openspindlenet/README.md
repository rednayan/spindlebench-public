# OpenSpindleNet (detector id `openspindlenet`)

Sejak M, Mivalt F, Sladky V, Vsiansky V, Carvalho DZ, St Louis EK, Worrell GA, Kremen V.
*OpenSpindleNet: An open-source deep learning network for reliable sleep spindle detection in scalp
and intracranial EEG.* Comput Biol Med 197 (2025) 110854. doi:10.1016/j.compbiomed.2025.110854,
PMC12499932. Preprint: medRxiv 10.1101/2025.04.13.25325696.

```
cd detectors/openspindlenet && uv sync          # Python 3.12, CPU onnxruntime
cd ../.. && detectors/openspindlenet/.venv/Scripts/python detectors/openspindlenet/run.py \
    --datasets dreams_spindles,nap --channels C3,F3,F4,C4,O1,O2,Cz
```

## Source and weights
| what | where |
|---|---|
| inference package (used here) | https://github.com/CaptainTrojan/openspindlenet, commit `ea012307c9b2a10a02099e172a5669e5ef5ec528` (4 Aug 2026), PyPI `openspindlenet==0.1.1` |
| research / training code (read, not run) | https://github.com/CaptainTrojan/mayo_spindles, commit `f090fb56ef038c3b27e3dab7adb3a680d438d59b` (the repository the paper cites) |
| weights | `openspindlenet/models/spindle-detector-eeg.onnx` (scalp model, md5 `f200501cafc2e0cd42b76e5766cdacc2`), shipped inside the wheel. The iEEG model (`spindle-detector-ieeg.onnx`) is not used. |
| licence | MIT (both repositories) |

## Pipeline (`run.py`)
Everything follows the authors' package (`openspindlenet/inference.py`) and their DREAMS export
(`mayo_spindles/dreams_to_hdf5.py`):
1. One channel at a time. Linear-interpolation resampling to **250 Hz** (as in their DREAMS export). No extra filtering: their DREAMS input was raw. NAP comes in already 0.1–45 Hz filtered from the cache.
2. Consecutive, non-overlapping **30 s blocks (7500 samples)** from t = 0, as described in the paper (Sec. 3.3.1). The leftover tail (< 30 s) is covered by one extra block aligned to the end of the recording. From that block, only detections centred in the uncovered tail are kept.
3. For each block:
   - z-score the raw block;
   - compute |CWT| with the `shan6-13` wavelet, 15 scales `geomspace(135, 270)` at 250 Hz (about 12–24 Hz), then z-score it;
   - run the ONNX model, which has two inputs (raw, scalogram) and a detection head of 30 one-second cells × (confidence, centre offset, duration).
4. The package's post-processing: sigmoid, confidence ≥ **0.5**, interval NMS at IoU **0.3**. The thresholds are the package defaults and were not tuned.
5. Output: `event = spindle`, `score =` detection confidence.

One deviation, which doesn't change the results: the CWT is computed with `method="fft"` and the blocks are batched. On float32 input this matches the package's direct-convolution CWT to about 2e-6 relative. The DREAMS outputs are byte-identical to the unmodified `SpindleInference.run()`, apart from two scores that differ in the 4th decimal. The input has to stay **float32**, as in the package and in the authors' training export, because pywt's float32 CWT differs from the float64 one by about 9%.

Units don't matter here: both inputs are z-scored per 30 s block.

## Runtime (14-core CPU, no GPU)
- DREAMS spindles, 8 × 30 min, one channel: 55 s total (about 7 s per 30-min channel).
  The unmodified package path took 157 s.
- NAP, 36 recordings of about 30 min × 7 channels (C3, F3, F4, C4, O1, O2, Cz) = 252 channel-recordings: 3955 s (66 min). That is about 16 s per channel, slowed by SSSM and other jobs sharing the CPU. On an idle machine expect about 7 s per channel, i.e. about 30 min.

## Sanity check (main venv, `experiments/rq1_benchmark/evaluate.py`, pooled F1, N2 scope)
`uv run python experiments/rq1_benchmark/evaluate.py --detectors openspindlenet,sssm,yasa_default --datasets dreams_spindles,dreams_kc,nap`

| dataset (reference) | protocol | P | R | **F1** | YASA defaults F1 |
|---|---|---|---|---|---|
| DREAMS spindles (expert 1, 0–1000 s) | onset ±0.5 s (main) | 0.43 | 0.88 | **0.57** (leaked, see caveat 1) | 0.42 |
| DREAMS spindles (expert 1) | IoU ≥ 0.2 | | | 0.59 | 0.43 |
| DREAMS spindles (union of experts, i.e. the training target) | onset ±0.5 s / IoU ≥ 0.2 | | | 0.64 / 0.66 | 0.37 / 0.38 |
| NAP C3, 30 included recordings (expert) | centre ±0.5 s (main) | 0.29 | 0.48 | **0.36** | 0.23 |
| NAP C3 | centre ±1.0 s | | | 0.49 | 0.32 |

- All stages: DREAMS F1 0.51, NAP F1 0.28 (NAP precision 0.20).
- Detection density on DREAMS is 7.4/min in N2 but still 5.4/min in wake and 5.1/min in REM (YASA defaults: 1.1 in N2, 1.0 in wake, 0.0 in REM). The model has almost no stage specificity (caveat 3), so give it N2-restricted input or score it in N2 only.
- The numbers are plausible. Against the union of experts (the model's training target) it reproduces the paper's 0.69 test-split F1 to within 0.03, so units, sampling rate and time offsets are correct.

## Caveats (read before putting numbers in the thesis)
1. **DREAMS is training data. These DREAMS numbers are leaked, not held-out.**
   - The scalp-EEG model was trained, validated and tested on DREAMS spindles only.
   - The authors split the 30 s segments of all 8 excerpts at random 70/15/15 (seed-dependent, not by subject; paper Sec. 3.1 and `dreams_to_hdf5.py`).
   - About 70% of the spindle-containing segments of every excerpt were seen in training, and the rest were used for model selection and testing.
   - The labels were the **union of expert 1 and expert 2**.
   - So `openspindlenet` on `dreams_spindles` is an optimistic, in-sample number. It must not be compared as a held-out result with the rule-based detectors. Use it only as a check that the wrapper is correct.
   - The paper reports a detection F1 of 0.69 on its DREAMS test split. Here it reaches about 0.66 against the union of experts (scored over the whole excerpt rather than their 15% test split), which confirms the wrapper reproduces the published model.
2. **NAP is out-of-distribution, so it is the only fair number.**
   - The model has never seen NAP, a montage other than C3-A1/Cz-A1, or a 200 Hz linked-mastoid recording.
3. **Trained only on 30 s segments that contain at least one spindle.**
   - Segments with no spindle, and artefacted segments (max/median |x| > 12), were excluded from training.
   - Combined with per-block z-scoring, this means the model has never learned to output "no spindle" for a whole quiet or wake block. It fires in every block.
   - Expect low precision on whole recordings, including wake and REM, which the N2-scoped rows partly hide.
4. **Fixed 30 s blocks.** A spindle that straddles a block boundary can be missed or split in two. The authors don't use overlapping blocks, so neither does this wrapper.
5. **Durations are bounded by the head's encoding.** Duration is predicted as `sigmoid × 2 s`, so detections are at most 2 s long.
6. The package's NMS drops any interval that starts exactly at sample 0 of a block. This is their padding convention and is kept as is.
