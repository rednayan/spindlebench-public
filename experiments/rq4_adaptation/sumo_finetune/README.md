# RQ4: fine-tuning SUMO on NAP (adaptation ladder)

Design: `docs/thesis/plan_rq4_finetuning.md`, label option A with half-width **±0.4 s** (chosen by
the user). Model: SUMO v1 (Kaulen et al. 2022) from its released MODA weights
(`../src/sumo/output/final.ckpt`), environment `../.venv-sumo` (Python 3.9, torch 1.10 CPU).
Detections go to `results/rq4/detections/<dataset>/<id>/<rec>.csv` (not the benchmark store).

## Run

```
P=detectors/sumo/.venv-sumo/Scripts/python.exe
$P detectors/sumo/finetune/prepare.py     # preprocess C3 / DREAMS, masks, folds.csv        (15 s)
$P detectors/sumo/finetune/baseline.py    # sumo_zeroshot / sumo_recal / sumo_persubj       (7 s)
$P detectors/sumo/finetune/run_all.py     # 50 training jobs, 11 parallel 1-thread workers  (80 min)
$P detectors/sumo/finetune/assemble.py    # detections, DREAMS fold averaging, curves PNG
uv run python experiments/rq1_benchmark/evaluate.py --store results/rq4/detections --datasets nap,dreams_spindles --out results/rq4/benchmark
.venv/Scripts/python.exe detectors/sumo/finetune/report.py   # tables + bootstrap CIs + learning_curve.png
```

Intermediate files (preprocessed signals, models, probabilities, histories, logs, `folds.csv`,
thresholds) are in `results/rq4/work/`. `train.py --fold k --rung r --seed s` runs one job.

## Data and splits

- 30 included naps / 17 subjects (`cache/manifest.csv`, `excluded == False`); the 6 excluded naps are
  never loaded (asserted in `prepare.py`). Channel C3 only.
- **Folds** (`results/rq4/work/folds.csv`, seed 0): subjects shuffled, split 4/4/3/3/3 into 5 test
  folds; both naps of a subject are always in the same fold. Within each fold's 13–14 non-test
  subjects, 2 random subjects are **validation** (early stopping + threshold), the remaining 11–12
  are **training** subjects in a fixed random order; the learning-curve subsets are the first
  1 / 3 / 6 / all of that order (nested). The validation subjects are the same for every rung of a fold.
- **Preprocessing** = the zero-shot adapter's `preprocess_sumo` (imported from `../run.py`):
  10th-order Butterworth HP 0.3 Hz + LP 30 Hz forward-backward, polyphase to 100 Hz, z-score over
  the **whole nap**. Training crops are cut from this whole-nap signal, so training and inference
  see identically normalised input.
- **Target**: per-sample mask = 1 within [mark − 0.4 s, mark + 0.4 s] (all marks, so a mark just
  outside an N2 run still shapes the mask at the crop edge).
- **Training data**: random 60-s crops whose start is uniform over all valid positions inside
  contiguous **expert-N2** runs ≥ 60 s of the training naps (N2 runs of 30 s are skipped: 3 runs,
  1.5 min in total). Training N2 per fold with all subjects: 244–304 min.
- **Inference**: whole held-out nap, stage-free, exactly as the zero-shot adapter.

## Rungs (ids)

| id | weights | threshold | notes |
|---|---|---|---|
| `sumo_zeroshot` | released | 0.5 (= published argmax) | identical to `results/detections/*/sumo` on all 38 recordings (checked in `baseline.py`: same events, onsets ≤ 0.01 s, scores ≤ 2e-4) |
| `sumo_recal` | released | best pooled N2 F1 on **all 13–14 non-test subjects** of the fold (train+val; the released model saw none of them) | |
| `sumo_persubj` | released | best F1 on the same subject's other nap | fallback to the fold's `sumo_recal` threshold for the 4 single-nap subjects (12, 14, 20, 25) and for suj19_d1, whose other nap has no N2 |
| `sumo_ft_n1/n3/n6/all` | fine-tuned from released | best val F1 of the kept epoch | |
| `sumo_scratch` | random init (seeded), all training subjects | same | same budget / early stopping as fine-tuning |
| `sumo_ft_all_s1/_s2`, `sumo_scratch_s1/_s2` | seeds 1, 2 | | seed changes crop sampling and (scratch) init; subsets/folds fixed |
| `sumo_ft_all_lr5e-3` | fine-tuned at lr 5e-3 | | lr sensitivity (= SUMO's and scratch's lr) |

Threshold grid: 0.05–0.95 in steps of 0.025; criterion = pooled F1, scope N2, detection centre vs
mark within ±0.5 s, one-to-one Hungarian matching (re-implemented in `ftlib.py` from
`experiments/rq1_benchmark/evaluate.py`/`spindle_eval.match`, since the main venv cannot be imported in the SUMO env); ties go
to the threshold closest to 0.5.

## Training choices (and why)

Taken from SUMO's own config (`src/sumo/config/default.yaml`, `sumo/model/sumo.py`):
GeneralizedDiceLoss on the dense logits (no moving average during training), Adam
(β 0.9/0.999, ε 1e-8, no weight decay), batch 12, BatchNorm in train mode, no augmentation,
model architecture unchanged. Changed:

| | SUMO (MODA) | here | reason |
|---|---|---|---|
| lr | 5e-3 | **1e-3 fine-tune**, 5e-3 scratch | pilot below; lower lr is standard for fine-tuning; 5e-3 kept for scratch = SUMO's recipe |
| input | 115-s MODA blocks, z-scored per block | 60-s random crops of the whole-nap z-scored signal | NAP N2 runs are 2–23 min; matches inference normalisation |
| epochs | 800, early stop patience 300 on val F1 (IoU) | "epoch" = 25 steps (300 crops = 5 h); max 100 epochs, patience 15 | CPU budget; fixed steps/epoch so every rung has the same budget |
| val metric | F1 at IoU thresholds on MODA blocks | best-threshold pooled N2 F1 at centre ±0.5 s on the 2 whole val naps-subjects, stage-free | = the evaluation protocol; NAP has no intervals |
| kept model | best val epoch | best val epoch **≥ 1** (epoch 0 = released weights is logged but not selectable) | so the fine-tuned rungs are actually fine-tuned |
| loop | Lightning Trainer | plain PyTorch loop, same loss/optimiser/forward | simpler on CPU/Windows; same computation |

**Learning-rate pilot** (`ft_all`, 30 epochs, folds 0 and 1, val subjects only; logs
`results/rq4/work/logs/pilot_*.log`; best val F1 within 30 epochs):

| lr | fold 0 | fold 1 |
|---|---|---|
| 5e-3 | 0.450 | 0.325 (very noisy: 0.17–0.33 epoch to epoch) |
| 1e-3 | 0.409 | 0.306 |
| 5e-4 | 0.375 (still rising) | 0.284 |
| 1e-4 | 0.242 | 0.283 |

I chose 1e-3 (5e-3 was best but unstable) and ran `sumo_ft_all_lr5e-3` on all folds as the
sensitivity check. The pilot uses validation subjects of folds 0/1, which are test subjects in
other folds; this is one scalar hyper-parameter, but it is a mild leak and is reported as such.
Two from-scratch pilots were stopped early (they informed nothing beyond "scratch learns within
~10–20 epochs").

## Results (experiments/rq1_benchmark/evaluate.py, NAP: centre ±0.5 s vs expert, 30 included naps, pooled F1)

`results/rq4/benchmark/rq4_nap_table.csv`; CI = recording-level paired bootstrap (2000) of the
difference to `sumo_zeroshot`; "better" = naps (of 29 with N2) where per-nap N2 F1 > zero-shot.

| id | F1 N2 | P / R (N2) | F1 all | ΔF1 N2 95% CI | better |
|---|---|---|---|---|---|
| sumo_zeroshot | 0.284 | 0.37 / 0.23 | 0.257 | | |
| sumo_recal | 0.284 | 0.37 / 0.23 | 0.257 | [−0.001, 0.000] | 0/29 |
| sumo_persubj | 0.284 | 0.37 / 0.23 | 0.257 | [−0.001, 0.000] | 0/29 |
| sumo_ft_n1 | 0.276 | 0.24 / 0.32 | 0.234 | [−0.064, +0.046] | 17/29 |
| sumo_ft_n3 | 0.324 | 0.29 / 0.37 | 0.270 | [−0.007, +0.097] | 14/29 |
| sumo_ft_n6 | 0.335 | 0.32 / 0.36 | 0.281 | [+0.017, +0.087] | 21/29 |
| sumo_ft_all | 0.320 | 0.30 / 0.34 | 0.282 | [−0.021, +0.089] | 21/29 |
| sumo_ft_all_s1 | 0.340 | 0.31 / 0.38 | 0.297 | [+0.004, +0.108] | 21/29 |
| sumo_ft_all_s2 | 0.336 | 0.31 / 0.37 | 0.290 | [+0.017, +0.094] | 22/29 |
| sumo_ft_all_lr5e-3 | 0.354 | 0.31 / 0.41 | 0.294 | [+0.019, +0.132] | 23/29 |
| sumo_scratch | 0.392 | 0.35 / 0.44 | 0.334 | [+0.032, +0.189] | 24/29 |
| sumo_scratch_s1 | 0.364 | 0.31 / 0.44 | 0.292 | [+0.014, +0.154] | 25/29 |
| sumo_scratch_s2 | 0.346 | 0.29 / 0.44 | 0.280 | [+0.000, +0.131] | 25/29 |

Seed spread (3 seeds): `ft_all` N2 0.320–0.340 (mean 0.332, sd 0.011), all 0.282–0.297;
`scratch` N2 0.346–0.392 (mean 0.367, sd 0.023), all 0.280–0.334.
References: SUMOv2 0.309, Mölle 2011 0.386, YASA rungs 0.24 / 0.35 / 0.39 (oracle).

**DREAMS forgetting check** (never trained on; onset ±0.5 s vs expert 1, 0–1000 s, pooled).
For each id the 5 fold models' per-sample p(spindle) are averaged (identical preprocessed input),
then thresholded at the mean of the 5 fold thresholds; `<id>_t05` thresholds at 0.5.

| id | F1 N2 | F1 all | P / R (all) |
|---|---|---|---|
| sumo (benchmark) = sumo_zeroshot | 0.620 | 0.561 | 0.83 / 0.43 |
| sumo_recal (mean NAP threshold 0.54) | 0.620 | 0.561 | |
| **sumo_ft_all** (mean thr 0.44) | **0.746** | **0.678** | 0.70 / 0.66 |
| sumo_ft_all_t05 | 0.748 | 0.682 | |
| sumo_ft_all_s1 / _s2 | 0.743 / 0.760 | 0.675 / 0.687 | |
| sumo_ft_all_lr5e-3 (mean thr 0.80) / _t05 | 0.637 / 0.712 | 0.570 / 0.642 | |
| sumo_scratch / _s1 / _s2 | 0.666 / 0.611 / 0.657 | 0.611 / 0.542 / 0.591 | |

(centre ±0.5 s rows are in `results/rq4/benchmark/rq4_dreams_table.csv`; they are 0.01–0.02 higher.)

## Training curves and runtime

`curves/<id>.png`: train loss and validation F1 per epoch for the 5 folds (solid = best threshold,
dotted = 0.5, dot = kept epoch); `curves/learning_curve.png`: F1 vs number of training subjects.
Per-fold thresholds, kept epochs and val F1: `results/rq4/work/training_summary.csv`.

Machine: Intel Core Ultra 5 225U (12 cores / 14 threads, 15 W laptop part, power-limited to
~1.1–1.5 GHz under full load). One training step (12 × 60 s) takes 0.3 s alone on one thread but
~1.7 s with 11 jobs in parallel. Wall time: prepare 15 s, baseline 7 s, LR pilot ~10 min,
`run_all.py` 80 min for 50 jobs (≈ 13 single-core hours), assemble + evaluate ~3 min.
Jobs stopped after 16–78 epochs (kept epochs 1–63).

## Caveats / things that look odd

- **Threshold rungs do nothing for SUMO.** Its moving-averaged logits make p(spindle) almost
  binary: from 1e-6 to 1 − 1e-5 the number of events on a nap changes by ~1–2. Recal/per-subject
  thresholds (which range from 0.05 to 0.875 across folds, a symptom of a flat F1 surface) change
  ≤ 4 detections on the whole dataset. The same saturation persists after fine-tuning (solid ≈
  dotted in the curves), so the fold thresholds of the fine-tuned models (0.075–0.95) are also
  close to arbitrary.
- **Validation is noisy.** 2 subjects (~100–300 marks) per fold; val F1 jumps by up to ±0.1
  between epochs (fold 1, fold 3), so the early-stopping epoch is partly luck (several folds keep
  epoch 1–5). The fold-3 validation set is suj19 (one nap has no N2) + suj25 (one nap).
- **From scratch beats fine-tuning** at the same budget (0.367 vs 0.332 N2, mean of 3 seeds), and
  fine-tuning at the higher lr 5e-3 (0.354) lands between them: the MODA initialisation does not
  help on NAP with these labels, and a larger lr (moving further from MODA) helps. Part of the
  difference may be the lr rather than the initialisation (scratch uses 5e-3).
- **Fine-tuning on NAP improves DREAMS** (0.62 → 0.75 N2) instead of causing forgetting: the
  released model is conservative on DREAMS (P 0.88 / R 0.48); NAP training raises recall. The
  ±0.4 s peak-centred targets fix event length at ~0.8 s; DREAMS onsets are true onsets, so onset
  matching slightly penalises the fine-tuned models (centre matching adds 0.01–0.02).
- **ft_n1** (one subject, 15–41 min of N2) is not better than zero-shot (0.276); the learning curve
  is flat from 3 subjects on, and n6 ≈ all within the seed spread.
- Excluded naps: none in training, validation, thresholds or test.
