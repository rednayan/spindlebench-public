# Changelog

Changes to protocols, benchmark definitions and results. Protocol versions are bumped whenever a
rule that changes F1 changes (see `src/spindlebench/protocols/`).

## Unreleased

### Restructure (branch `restructure`)
- Core code moved into the `spindlebench` package (`src/spindlebench/`): event matching
  (`metrics.events`), point-mark scoring (`protocols.nap_points`, version `nap_points.v1`), NAP loaders
  (`data.nap`, `data.nap_annotations`), paths (`paths`). Every re-run result is byte-identical to the
  2026-10-28 snapshot.
- Detectors (phase 2): `detectors/<name>/manifest.json` for all 8 adapters (ids, method, upstream
  source and licence, adapter licence, weights, environment, lock, command); `spindlebench.detections`
  holds the detection file contract; `tests/test_detectors.py` checks manifests, id ownership and every
  stored detection file. `detectors/wonambi/uv.lock` added (matches the environment that produced the
  results). Detection runs and experiments now write provenance records (`spindlebench.provenance`,
  `detectors/common.py`). A re-run of `moelle` on DREAMS is byte-identical to the stored detections.
- DREAMS assessment rewritten (`protocols.dreams_assessment`, `dreams_assessment.v1`) from the published
  description of Devuyst et al. (2011) and our notes on its behaviour, replacing the literal port of the
  DREAMS MATLAB code (distributed under CC BY-NC-ND 3.0). Confirmation (`tests/test_dreams_assessment.py`):
  identical to the original MATLAB counts for the shipped automatic detections in all five comparisons
  and to the published 70.20% / 26.44%; identical to the old port on 107 of 140 detector rows, the rest
  within 4 events (F1 within 0.003). Causes, found by running the old port as a black box: it counts a
  detection starting at 0 s twice (A7, SSSM), and it handles clusters linked only through a third scorer
  in a way the description does not define.
- NAP detection outputs may be published (decision 2026-10-28; NAP is BSD 3-Clause on OSF).
- Release hygiene (phase 4): `CITATION.cff` (validated); REUSE 3.3 compliant licensing
  (`LICENSES/`, `REUSE.toml`; code MIT, docs and results CC BY 4.0, two GPL-3.0 ports with SPDX
  headers); pre-commit hooks (large files, merge conflicts, TOML/YAML/JSON, private keys, whitespace,
  ruff); CI on Linux and Windows (lint, tests) plus REUSE and citation checks; `docs/development.md`.
  Every experiment script records provenance in `results/provenance/<study>/<script>.json`
  (`provenance.track`), checked by a test.
- Experiments (phase 3): every top-level script moved to `experiments/<study>/` (rq1_benchmark,
  rq2_validity, rq3_channels, rq4_adaptation, raters, nap_scoring, injection, label_free, yasa_sweeps,
  usability); the SUMO fine-tuning and injection-training code moved there from `detectors/sumo/`.
  Protocol definitions and cache readers moved into the package (`protocols.benchmark`, version
  `benchmark.v1`; `data.cache`); the cache export and downloaders are `spindlebench cache` and
  `spindlebench download`. All paths come from `spindlebench.paths`; scripts and adapters run from any
  working directory. All 24 analyses re-run: every result identical to the snapshot.
- Cache manifests store signal paths relative to the manifest with forward slashes (portable); readers
  accept the old form too. Re-exported labels and DREAMS signals are identical.
- `spindlebench cache` now keeps BOAS in a full export (it was dropped unless `--boas-only` ran after).
- PhysioNet downloads fetch files from the AWS open-data mirror (checksums still from physionet.org).
- Provenance records replace local paths with `<root>` and `~`. SUMO runtimes moved from
  `detectors/sumo/runtime.csv` to `results/provenance/detections/sumo_runtime.csv`.
- Generated tables: `scripts/make_tables.py` -> `results/results_tables.md`.
- Retired: `run_yasa.py`, `test_dreams.py`, `check.py`, `main.py` (exploration only). DREAMS
  distribution files (`readme.txt`, `global_assessmentMethod.m`, licence) are no longer tracked.
- Licences found: `detectors/a7/a7.py` and `detectors/kcomplex_rules/spinky.py` are ports of GPL-3.0
  code and are GPL-3.0; `dreams_auto` detections are part of the DREAMS database (CC BY-NC-ND 3.0) and
  must not be published; SSSM weights have no licence (not redistributed).
- `affiliation` (Huet et al. 2022) is now a locked dependency pinned to commit 8d84498 instead of a
  manual clone in `third_party/`.

### Results
- `results/validity/greedy_vs_optimal*.csv` regenerated. The stored tables predated the A7 detector
  change of 2026-10-27 (`4337636`) and lacked four later detector configurations. Conclusion unchanged:
  greedy and maximum matching differ in 3 comparisons, by one true positive each, at ±1.0 s.
- `matching_greedy_vs_optimal.py` skips label-free recordings (BOAS), which made it fail.
