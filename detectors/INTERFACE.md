# Detector interface

Every detector, rule-based or learned, is wrapped so that it reads the **same
cached input** and writes the **same output format**. One evaluator
(`experiments/rq1_benchmark/evaluate.py`) then scores every detector under every protocol (RQ1, RQ2).

## Input: `cache/` (built by `uv run spindlebench cache`)

`cache/manifest.csv` has one row per recording:

| column | meaning |
|---|---|
| `dataset` | `dreams_spindles` (8), `dreams_kc` (10), `nap` (36) |
| `rec` | recording id, e.g. `excerpt3`, `suj13_d1` |
| `path` | `.npz` file with `data` (n_ch × n_samples, float32, **µV**), `sf`, `ch_names`; relative to the manifest's folder, forward slashes (resolve with `common.signal_path`) |
| `sf` | sampling rate (DREAMS: native 50/100/200 Hz; NAP: 200 Hz) |
| `default_channel` | the scored channel (DREAMS: C3-A1 or CZ-A1; NAP: C3) |
| `events` | which event types are labelled there (`spindle`, `kcomplex`) |

Preprocessing already applied is minimal: DREAMS is untouched; NAP is
linked-mastoid referenced, 0.1–45 Hz, resampled to 200 Hz. **Apply the
detector's own published preprocessing on top** (resampling, filtering,
z-scoring...) inside the adapter, and document it.

## Output: `results/detections/<dataset>/<detector>/<rec>.csv`

```
onset,duration,channel,event,score
412.37,0.84,C3,spindle,0.91
```

- `onset`, `duration` are in seconds from the start of the recording.
- `event` is `spindle` or `kcomplex`.
- `score` is optional (a probability or confidence); leave it empty if there isn't one.
- `<detector>` is a short id. Use one id per configuration, e.g. `sumov2`,
  `sumo_moda`, `seed`, `a7`, `yasa_default`.
- Write a file for every recording you ran, even if it has no events (header only).

## Which recordings and channels

- `dreams_spindles`: all 8, on the single channel (spindle detectors).
- `dreams_kc`: all 10, on the single channel (K-complex detectors).
- `nap`: all 36, at least on `C3`. If runtime allows, also run
  `F3 F4 C4 O1 O2 Cz`. These are used for the channel-count ablation (E3).
  NAP labels both spindles and K-complexes.

## Provenance

The first file each run writes for a (dataset, detector) also writes
`results/provenance/detections/<dataset>/<detector>.json` (done by `common.write`): time, command line,
git commit and whether the tree had uncommitted changes, Python and package versions, and the
arguments. Experiments write the same record with `spindlebench.provenance.write_provenance`.

## Manifest: `detectors/<name>/manifest.json`

Every adapter declares what it is, so the benchmark can be audited and cited without reading code:

| key | meaning |
|---|---|
| `ids` | every detector id the adapter writes, with a one-line description |
| `events`, `kind` | `spindle` / `kcomplex`; `rule-based` / `learned` |
| `method` | the paper (citation, DOI) |
| `source` | upstream code: URL, commit, licence, and how the adapter relates to it (port, unmodified call, written from the paper) |
| `adapter_licence` | licence of the adapter code itself (GPL-3.0 for ports of GPL code, MIT otherwise) |
| `weights` | learned models: file, checksum, licence |
| `environment` | `main` (the core environment) / `uv-project` / `venv-requirements`, Python version, lock file |
| `command` | how to run it |

`tests/test_detectors.py` checks the manifests (every adapter has one, ids unique, lock files exist),
that every stored detector id has an owner, and that every stored detection file follows the format.

## Adapter layout

```
detectors/<name>/
  run.py        python run.py --manifest cache/manifest.csv --out results/detections
                              [--datasets nap,dreams_spindles] [--channels C3,Cz] [--recs suj13_d1]
  manifest.json machine-readable summary (above)
  README.md     source repo + commit, weights + licence, preprocessing, runtime, caveats,
                sanity check result
  pyproject.toml / .venv   own environment if its dependencies clash with the main one
```

The main project venv is Python 3.14 (`.venv`, mne + yasa). Learned detectors
usually need an older Python/PyTorch, so create a separate uv environment
inside `detectors/<name>/`, e.g. `uv venv --python 3.11`. CPU only: there is no
CUDA GPU on this machine (14 cores, 31 GB RAM).

## Sanity check before handing over

Score your detector on `dreams_spindles` against expert 1 (window 0–1000 s,
onset ±0.5 s) with `uv run python experiments/rq1_benchmark/evaluate.py --detectors <name> --datasets dreams_spindles`.
Compare it with the reference rows (YASA defaults pooled F1 0.40, YASA loosened
0.57, DREAMS automatic 0.62, expert 2 0.65). A number far below YASA defaults
usually means wrong units, a wrong sampling rate, or wrong time offsets.
