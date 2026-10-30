# Experiments

One folder per study. Each script is run from anywhere with `uv run python experiments/<study>/<script>.py`,
reads the cache (`uv run spindlebench cache`) and the stored detections (`results/detections/`, written by
`detectors/<name>/run.py`), and writes to `results/`. Paths come from `spindlebench.paths`; set
`SPINDLEBENCH_DATA`, `SPINDLEBENCH_CACHE` or `SPINDLEBENCH_RESULTS` to move them.

Run order inside a study is top to bottom. The table at the end of the main README maps thesis and paper
items to the commands that produce them.

| study | question | scripts (in order) | main outputs |
|---|---|---|---|
| `rq1_benchmark` | How do detectors compare on DREAMS and NAP under one protocol? | `evaluate.py`, `dreams_official.py` | `results/benchmark/` |
| `rq2_validity` | Does the ranking survive the evaluation choices (rule, tolerance, reference, matching)? | `validity_e2.py`, `protocol_curves.py`, `rank_transfer.py`, `matching_greedy_vs_optimal.py`, `kc_anchor_sensitivity.py` | `results/validity/`, `results/sensitivity/kc_anchor.csv` |
| `rq3_channels` | What changes the result: channel count, sleep stage, montage? | `ablation_e3.py detect`, `ablation_e3.py analyse`, `ablation_e3_stage.py`, `montage_ablation.py build`, `montage_ablation.py detect` (then `evaluate.py --store ...`) | `results/e3/`, `results/sensitivity/montage/` |
| `rq4_adaptation` | What is the cheapest adequate adaptation to NAP? | `adaptation_nap.py`, `adaptation_moelle.py`, `sumo_finetune/` (SUMO environment, see its README), `adaptation_ladder.py`, `rq4_label_budget.py` | `results/adaptation/`, `results/rq4/` |
| `raters` | How well do human experts agree (the ceiling)? | `interrater.py` (DREAMS), `mass_raters.py` (MASS SS2 annotations), `nap_rater_rule.py` (NAP rater's multichannel rule) | `results/validity/interrater.csv`, `results/mass_raters/`, `results/nap_scoring/rater_rule.csv` |
| `nap_scoring` | How should detections be scored against point marks? | `nap_scoring.py` (protocol `nap_points.v1`) | `results/nap_scoring/` |
| `injection` | SpindleInject: known ground truth for recall by strength; injection training | `build_bench.py`, `run_bench_detectors.sh`, `score_bench.py`, `realism.py`, `slow_spindles.py`, `sumo_inject_ft/` (SUMOv2 environment) | `results/injection/` |
| `label_free` | Do detectors agree on a dataset without spindle labels (BOAS)? | `boas_labelfree.py` | `results/boas/` |
| `yasa_sweeps` | YASA threshold sweeps (feed `yasa_dreams_tuned`) and NAP data description | `dreams_pipeline.py`, `nap_pipeline.py` | `results/dreams/`, `results/nap/` |
| `usability` | Runtime of each detector adapter | `usability_timing.py` | `results/usability/runtime.csv` |

Tables for the thesis: `uv run python scripts/make_tables.py` → `results/results_tables.md`.

## Rules for new experiments

- A new study gets its own folder with a short README (question, scripts, outputs, paper item).
- Paths only from `spindlebench.paths`; no paths relative to the working directory.
- Call `provenance.track(__file__, protocols=[...])` after the imports (writes `results/provenance/<study>/<script>.json`).
- A change to a matching rule, scope or data cut gets a new protocol version and a `CHANGELOG.md` line.
