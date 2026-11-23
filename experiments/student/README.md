# Student: a label-free spindle detector trained on detector consensus

Question, claims, architecture and the full protocol: `notes/plans/student_architecture.md` (private
companion repository). This README lists what has been run and the commands, in order.

No expert spindle label is used for training, validation or selection. Training sites are HMC and the
CinC 2018 subset (public hypnograms only). DREAMS and NAP are the gold sets: `data.guard` refuses them
for every training-side call. Before the freeze, the gold sets are touched only by `label_models.py gold`,
which reads the teachers' outputs there (never an expert label), and by `evaluate.py`, which scores the
consensus rows.

## Environment

Its own uv project, Python 3.14 (the benchmark's version; torch, crowd-kit and FlyingSquid all install
on it), with `spindlebench` as an editable path dependency:

    cd experiments/student && uv sync                                  # CPU torch (default)
    cd experiments/student && uv sync --no-group cpu --group cuda      # NVIDIA GPU: torch 2.14.1+cu126

The cpu group pins torch 2.15.0 (the CPU grid's runs used `2.15.0+cpu`); the cuda group pins 2.14.1+cu126, the
newest torch with a CUDA 12 wheel for Python 3.14 on Windows (the GPU machine, an RTX 4070, has a CUDA 12.9
driver; 2.15.0 ships CUDA 13 only). The student code gives the same losses under both versions on CPU. On the GPU machine pass
`--no-group cpu --group cuda` to every `uv run` as well, or uv switches back to the CPU wheel.

Run folders live outside the repository, in `../runs/student/<run>/` next to the checkout (override with
`SPINDLEBENCH_RUNS`); every script reads them from `data.RUNS`. Older runs kept as evidence are in
`../runs/archive/`: `collapsed_lam1/`, `superseded_guard1/`, and `interrupted/` (the 10 runs stopped when
the launcher was killed on 2026-11-05: metrics and configs only, no checkpoints; they were rerun from
scratch, because a pass checkpoint holds the weights but not the optimiser or sampler state).

## GPU

`train.py --device cpu|cuda` and `run_all.py --device cpu|cuda` choose where a run trains; `--track`
still chooses the settings (cpu: trunk S, batch 12, seeds 0–2; gpu: trunk M, batch 32, seeds 0–4). Each
track defaults to its own device. Runs of the gpu track are named `<arm>[_<tag>]_gpu_s<seed>`, so they never
collide with the CPU grid. On CUDA, training is deterministic per seed (`CUBLAS_WORKSPACE_CONFIG`,
deterministic algorithms), but not bit-identical to the same seed on a CPU; `config.json` records device,
GPU, host and CPU, so mixed-device results can be reported as such. CPU and CUDA jobs have separate worker
limits (`--workers`, default 10 on cpu, 3 on cuda), so two launchers can share one machine.

First time on the GPU machine (from `experiments/student`):

    uv run --no-group cpu --group cuda pytest ../../tests/test_student_model.py     # includes a CPU-vs-CUDA loss check
    uv run --no-group cpu --group cuda python train.py --arm student_rules4 --seed 0 --device cuda --max-batches 20 --no-val --tag smoke

then delete `../runs/student/student_rules4_smoke_s0/`.

Unattended GPU track: `run_gpu_grid.cmd [workers]` (Windows) trains every `run_all.py --track gpu` job
(65: the CPU grid's 13 configurations, trunk M, all training data, 5 seeds) and then runs `selection.py
score` on every finished run. Run it again after a reboot: finished runs are skipped, interrupted ones are
archived to `../runs/archive/interrupted/` and rerun. The plan's remaining GPU arms (leave-one-teacher-out,
SUMOv2 as fifth teacher, data scale, ...) are not in the job list yet.

FlyingSquid (0.0.0a0, 2020) imports `pgmpy.models.MarkovModel`, so pgmpy is pinned below 0.1.13, and
that pgmpy calls `np.product` (removed in numpy 2): `label_models.py` adds that one alias before the import.

## Step 1: teachers, data, targets

| # | command | output |
|---|---|---|
| 0 | `uv run spindlebench artefacts` (repo root) | `cache/labels/<ds>/<rec>__artefact.csv`, `results/artefacts/` |
| 1 | `bash results/student/logs/run_teachers_clean.sh` (repo root; restartable) | `results/detections_clean/<ds>/<teacher>/` |
| 2 | `uv run python splits.py` | `results/student/work/splits.csv` |
| 3 | `uv run python prepare.py` | `results/student/work/prep/` (4.2 GB, not tracked) |
| 4 | `uv run python rasterise.py` | `results/student/work/teachers/`, `results/student/teachers/*.csv` |
| 5 | `uv run python label_models.py fit` | `results/student/work/targets/label_models.json`, `results/student/teachers/label_models.csv` |
| 6 | `uv run python label_models.py gold` | `results/student/detections/<gold>/consensus_*/` |
| 7 | `uv run python experiments/rq1_benchmark/evaluate.py --store results/student/detections --out results/student/consensus --datasets dreams_spindles,nap` (repo root) | `results/student/consensus/` |
| 8 | `uv run python experiments/rq1_benchmark/evaluate.py --store results/detections_clean --out results/student/consensus_teachers --datasets dreams_spindles,nap --detectors a7,luna_spindles,wonambi_ucsd,yasa_default` (repo root) | `results/student/consensus_teachers/` |

Teachers: `a7`, `luna_spindles`, `wonambi_ucsd`, `yasa_default`, each run stage-free on the
artefact-free windows of the default channel (`detectors/<name>/run.py --clean`).

## Step 2: model, losses, decoders, identifiability, pilot

Architecture, equations and diagrams: [`docs/student_architecture.md`](../../docs/student_architecture.md).

| # | command | output |
|---|---|---|
| 9 | `uv run python build_val_injection.py` | `cache_student_inj/v0..v3/`, `results/student/injection/v*/events/` |
| 10 | `bash results/student/logs/run_teachers_inj.sh` (repo root; restartable) | `results/student/injection/v*/detections/` |
| 11 | `uv run python score_val_injection.py teachers` | `results/student/injection/teacher_recall.csv`, `identifiability.json` |
| 12 | `uv run python train.py --arm student_soft --seed 0` (pilot) | `../runs/student/student_soft_s0/` (was `results/student/work/runs/`, committed in 36b9f89e) |
| 13 | `uv run python figures.py` | `docs/figures/student/stream_b_init.png` |
| 14 | `uv run python visual_guide.py` (regenerate whenever results change) | `docs/visual_guide.html` |

Tests: `uv run --project experiments/student pytest tests/test_student_model.py tests/test_student.py`
(repo root).

Timing on the laptop (timed 200-batch runs, 2026-11-04, one thread, CPU track, batch 12, every loss
term on): `student_rules4` 0.59 s per batch (32 min per pass of 3,241 batches); `student_soft` 0.26 s
(14 min per pass). Validation on the 40 nights: about 11 min per pass. A 20-pass run is therefore at
most ≈ 14 h (rules4) or ≈ 8 h (plain); the 42 CPU-track runs ≈ 460 CPU-hours, ≈ 2 days on 11 workers.

Pilot result (2026-11-04): early stop after pass 8, Borda selects pass 3, F1 against the majority vote on
the validation nights 0.703 at τ = 0.5 (gate ≥ 0.70 passed); a diagnostic τ sweep gives 0.754 at τ = 0.3.

## Step 3: CPU-track grid, selection, freeze, one gold-set run

| # | command | output |
|---|---|---|
| 15 | `uv run python run_all.py --track cpu [--device cuda] [--workers 10]` (restartable) | `../runs/student/<run>/`, logs `results/student/logs/train_<run>.out` |
| 16 | `uv run python selection.py score --workers 10` (every finished run) | `results/student/selection/scores/` |
| 17 | `uv run python selection.py freeze [--track cpu]`, then commit `frozen.json` | `results/student/selection/frozen.json`, `summary.csv` |
| 18 | `uv run python infer.py` (refuses without a committed freeze) | `results/student/detections/<gold>/<run>/` |
| 19 | `uv run python experiments/rq1_benchmark/evaluate.py --store results/student/detections --out results/student/benchmark --datasets dreams_spindles,nap` (repo root) | `results/student/benchmark/` |
| 20 | `uv run python experiments/student/report.py` (repo root; NAP `nap_points.v1` scoring, hypotheses H1–H5) | `results/student/report/*.csv`, `report.md` |
| 21 | `uv run python experiments/student/report_figures.py` (repo root, after 20) | `results/student/report/figures/` |

The compiled GPU-track results, with every figure and table, start at
[`results/student/report/README.md`](../../results/student/report/README.md).

CPU grid (trunk S, seeds 0–2, 39 runs): `student_rules4` (λ 0.01, amendment 8); `student_soft`, `_mv`, `_ds`; the four
per-teacher students; ablations of `student_rules4`: λ 0, λ 0.1, stream B off, global matrices, head 2 off.
The plan's CPU ablation "stream B into the trunk" is moved to the GPU track (it would change SUMOv2's
network, which trunk S must not do); the global-matrix ablation takes its place. The SUMOv2-as-fifth-teacher
arm (3 runs) follows once SUMOv2 has been run on the corpus.

Amendment 8 (2026-11-04, evening). The first grid used trace weight λ = 1, summed over teachers, and every
run with a noise branch collapsed at pass 3 (label swap; `docs/student_architecture.md`, section 15). Those
seven runs are kept in `../runs/archive/collapsed_lam1/` as a negative result. The grid now uses
Tanno et al.'s λ = 0.01 with the trace averaged over teachers. `train.py` stops a run whose probability-weighted mean
confusion matrix is no better than chance, diag0 + diag1 ≤ 1 (`done.json`: `collapsed`, no selected pass);
diagonal dominance is logged per pass (`dominant`). `run_all.py` never runs more than `--workers`
trainings on the machine, counting runs started by any launcher.

Amendment 9 (2026-11-05, before any selection). After warm-up the trace term pushed the λ = 0.01 and 0.1
students towards over-detection (pass 4 at λ = 0.01: N2 4.7 per min, against 2.6 for the majority vote).
λ ∈ {0, 0.01, 0.1} are now all headline candidates, and `selection.py freeze` picks the λ with the highest
mean injected F1 over its seeds (ties: fewer induced false alarms). The agreement Borda still picks each
run's pass but not λ, because it favours the smallest λ by construction.

Amendment 10 (2026-11-09, after the GPU grid was scored, before any freeze and before any gold-set run).
Three changes, decided on the label-free validation scores only:
1. Decoder. The injection criterion chose the lowest threshold of the grid, τ = 0.15, for every run of
   every arm, and ranked runs by how much they over-detect: at their chosen settings `rules4_lam0` came
   first (injected F1 0.524, precision 0.57, N2 7.1 per min) and the majority-vote student last but one
   (0.491, precision 0.80, N2 4.2). Injected F1 spans only 0.49–0.52 across arms, while precision and
   false alarms differ widely. The criterion averages F1 over SNR levels down to −3 dB, where recall is
   bought by lowering the threshold. Every run is now frozen at D0, τ = 0.5, no snapping: the decision
   rule every student was trained for (its output estimates P(spindle), so 0.5 is the decision
   between the two classes) and SUMOv2's own. The injection scores are reported for every run, and the
   setting the old criterion would have chosen is kept in `frozen_<track>.json` as `injection_choice`.
2. Headline. `student_rules4` fixA replaces the λ choice of amendment 9: the noise tables start at each
   teacher's Dawid–Skene confusion matrix and the per-sample correction is bounded by ±0.5 logit, λ = 0
   (rules4-fix pilot and full runs, `CHANGELOG.md`). All five seeds kept diagonal dominance through training;
   every unanchored `student_rules4` run lost it after warm-up.
3. Collapsed families. All five λ = 0.1 seeds were stopped by the label-swap guard (amendment 8). A
   family without a surviving run is reported as failed (seeds 0) and is never selected. The λ table of
   amendment 9 is reported at the fixed setting and decides nothing.
