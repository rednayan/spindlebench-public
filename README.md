# spindlebench

A benchmark for sleep spindle and K-complex detectors, built to show how much a detector's score
depends on the reference it is scored against.

It runs rule-based and learned detectors on public sleep EEG under one scoring code, with every
protocol choice explicit and versioned:
- DREAMS, NAP, MASS SS2 annotations, BOAS, HMC and Sleep-EDF;
- matching rule, tolerance, reference annotator, sleep-stage scope;
- point-mark scoring;
- injected ground truth (SpindleInject).

Status: research code for a Master's thesis (THWS, 2026–27) and a paper in preparation. The interfaces
may still change; every change to a scoring protocol is logged in [`CHANGELOG.md`](CHANGELOG.md).

## Install

```bash
uv sync                  # core environment: Python 3.14, the spindlebench package, YASA, MNE
uv run pytest            # 136 tests, a few seconds
uv run spindlebench paths
```

Learned detectors have their own environments because their dependencies conflict. Each is
described by `detectors/<name>/manifest.json` and set up as in its README:
- OpenSpindleNet and SSSM: `uv sync` in their folder;
- SUMO / SUMOv2: two pinned virtual environments;
- wonambi: `uv sync` in its folder.

## Data

Raw data are never in this repository. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for each dataset's
licence, citation and what may be shared.

```bash
uv run spindlebench download nap        # OSF chav7, ~18.5 GB
uv run spindlebench download boas       # OpenNeuro ds005555 subset
uv run spindlebench download hmc        # PhysioNet, checksums verified
uv run spindlebench download sleepedf   # PhysioNet sleep-cassette, checksums verified
# DREAMS (Zenodo 2650142) and MASS (Borealis) require manual download; see DATA_SOURCES.md
uv run spindlebench cache               # one standard format for every detector
```

Locations default to the repository folder. Set `SPINDLEBENCH_DATA`, `SPINDLEBENCH_CACHE` or
`SPINDLEBENCH_RESULTS` to put them elsewhere.

## Run

```bash
uv run python detectors/yasa/run.py --datasets dreams_spindles,nap   # one adapter; see detectors/
uv run python experiments/rq1_benchmark/evaluate.py                  # score everything
uv run python scripts/make_tables.py                                 # results/results_tables.md
```

[`experiments/README.md`](experiments/README.md) lists every study, its scripts in run order and its
outputs. Every run writes `results/provenance/<study>/<script>.json`, which records:
- the code version, and whether there were uncommitted changes;
- the package versions;
- the settings and protocol version.

## Layout

```
src/spindlebench/   the package: paths, data loaders and cache export, matching and metrics,
                    scoring protocols (versioned), detection format, provenance, SpindleInject generator
detectors/          one adapter per detector, each with a manifest; own environment where needed
experiments/        one folder per study (rq1-rq4, raters, nap_scoring, injection, label_free, ...)
scripts/            tables, snapshot comparison
results/            small result tables and figures (large outputs are not in git)
tests/              metric goldens, detector conformance, script checks
docs/               protocol decisions, development rules
```

## Reproducing published numbers

The pipeline reproduces two published results exactly:
- **Devuyst et al. 2011**, through the DREAMS assessment method rewritten from its description
  (`experiments/rq1_benchmark/dreams_official.py`): sensitivity 70.20%, false positives 26.44%;
- **SUMOv2 on DREAMS** (`detectors/sumo/check_reproduction.py`): published 0.53 / 0.76, here 0.532 / 0.764.

A table mapping each number in the paper to its command will be added with the paper.

## Licence

- **Code:** MIT ([`LICENSE`](LICENSE)).
- **GPL-3.0 exceptions:** two detector adapters are ports of GPL-3.0 code and are GPL-3.0:
  - `detectors/a7/a7.py` (A7, Lacourse et al. 2019);
  - `detectors/kcomplex_rules/spinky.py` (Spinky, Lajnef et al. 2017).
- **Documentation and released data:** CC BY 4.0.
- **Third-party datasets and weights:** they keep their own licences ([`DATA_SOURCES.md`](DATA_SOURCES.md)).
